import itertools
import math
import os
import pickle
import random
from argparse import Namespace, ArgumentParser, FileType
from functools import partial
import copy
from rdkit.Chem import RemoveHs

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm

from datasets.pdbbind_affinity_davis import PDBBind
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split, all_atoms):
    cache_path = args.cache_path
    if not args.no_torsion:
        cache_path += '_torsion'
    if all_atoms:
        cache_path += '_allatoms'
    split_path = args.split_train if split == 'train' else args.split_val
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                       + ('' if args.no_torsion or args.num_conformers == 1 else
                                           f'_confs{args.num_conformers}')
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings'))
    return cache_path

def get_args_and_cache_path(original_model_dir, split):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args, get_cache_path(model_args,split)

def get_model_args(original_model_dir):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class AffinityDataset(Dataset):
    def __init__(self, batch_size, cache_path, original_model_dir, confidence_model_dir, confidence_ckpt, split, device, limit_complexes,
                 inference_steps, samples_per_complex, no_random, ode, no_final_step_noise, all_atoms,
                 args, balance=False, use_original_model_cache=True, rmsd_classification_cutoff=2,
                 cache_ids_to_combine= None, cache_creation_id=None, heterographs_name=None, heterographs_split_size=None, heterographs_combine=None):

        super(AffinityDataset, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        # self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.no_random = no_random
        self.ode = ode
        self.confidence_model_dir = confidence_model_dir
        self.confidence_ckpt = confidence_ckpt
        self.no_final_step_noise = no_final_step_noise
        self.actual_steps = None
        self.heterographs_name = heterographs_name
        self.heterographs_split_size = heterographs_split_size
        self.heterographs_combine = heterographs_combine
        
        if args.confidence_model_dir is not None:
            with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
                self.confidence_model_args = Namespace(**yaml.full_load(f))
    
        self.original_model_args = get_model_args(original_model_dir)
        self.complex_graphs_allatom_cache = get_cache_path(args, split, all_atoms=True)
        self.complex_graphs_calpha_cache = get_cache_path(args, split, all_atoms=False)
    
        #print('Using the cached complex graphs of the original model args' if self.use_original_model_cache else 'Not using the cached complex graphs of the original model args. Instead the complex graphs are used that are at the location given by the dataset parameters given to confidence_train.py')
        # print(self.complex_graphs_cache)
        
        # the original calpha complex graphs for scoring model
        if not os.path.exists(os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl")):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                    receptor_radius=args.receptor_radius,
                    cache_path=args.cache_path, split_path=args.split_val if split == 'val' else args.split_train,
                    remove_hs=args.remove_hs, max_lig_size=args.max_lig_size,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                    matching=not args.no_torsion, keep_original=True,
                    popsize=args.matching_popsize,
                    maxiter=args.matching_maxiter,
                    all_atoms=False,
                    atom_radius=args.atom_radius,
                    atom_max_neighbors=args.atom_max_neighbors,
                    esm_embeddings_path=args.esm_embeddings_path,
                    require_ligand=True)
            
        if self.heterographs_split_size:
            with open(os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl"), 'rb') as f:
                self.complex_graphs_calpha = pickle.load(f)
                
            batch_size = args.heterographs_split_size
            for ind, x in enumerate(batch([i for i in range(len(self.complex_graphs_calpha))], batch_size)):
                heterographs_chunk = [self.complex_graphs_calpha[i] for i in x]
                with open(os.path.join(self.complex_graphs_calpha_cache, f'heterographs_{ind}.pkl'), 'wb') as f:
                    pickle.dump(heterographs_chunk, f)
                    
        
        
        # for splitting heterographs
        # if self.heterographs_name:
        #     print(f'HAPPENING | Loading calpha complex graphs from: {os.path.join(self.complex_graphs_calpha_cache, f"{self.heterographs_name}.pkl")}')
        #     with open(os.path.join(self.complex_graphs_calpha_cache, f"{self.heterographs_name}.pkl"), 'rb') as f:
        #         self.complex_graphs_calpha = pickle.load(f)
        # else:
        #     print(f'HAPPENING | Loading calpha complex graphs from: {os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl")}')
        #     with open(os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl"), 'rb') as f:
        #         self.complex_graphs_calpha = pickle.load(f)
        # self.complex_graph_calpha_dict = {d.name: d for d in self.complex_graphs_calpha}
        
        # the original allatom complex graphs for confidence model
        if not os.path.exists(os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl")):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                    receptor_radius=args.receptor_radius,
                    cache_path=args.cache_path, split_path=args.split_val if split == 'val' else args.split_train,
                    remove_hs=args.remove_hs, max_lig_size=args.max_lig_size,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                    matching=not args.no_torsion, keep_original=True,
                    popsize=args.matching_popsize,
                    maxiter=args.matching_maxiter,
                    all_atoms=True,
                    atom_radius=args.atom_radius,
                    atom_max_neighbors=args.atom_max_neighbors,
                    esm_embeddings_path=args.esm_embeddings_path,
                    require_ligand=True)
            
        if self.heterographs_split_size:
            with open(os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl"), 'rb') as f:
                self.complex_graphs_allatom = pickle.load(f)
                
            batch_size = args.heterographs_split_size
            for ind, x in enumerate(batch([i for i in range(len(self.complex_graphs_allatom))], batch_size)):
                heterographs_chunk = [self.complex_graphs_allatom[i] for i in x]
                with open(os.path.join(self.complex_graphs_allatom_cache, f'heterographs_{ind}.pkl'), 'wb') as f:
                    pickle.dump(heterographs_chunk, f)
            assert False
            
        if self.heterographs_split_size:
            assert False
        
        if self.heterographs_name:
            print(f'HAPPENING | Loading allatom complex graphs from: {os.path.join(self.complex_graphs_allatom_cache, f"{self.heterographs_name}.pkl")}')
            with open(os.path.join(self.complex_graphs_allatom_cache, f"{self.heterographs_name}.pkl"), 'rb') as f:
                self.complex_graphs_allatom = pickle.load(f)   
        else:
            print(f'HAPPENING | Loading allatom complex graphs from: {os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl")}')
            with open(os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl"), 'rb') as f:
                self.complex_graphs_allatom = pickle.load(f)
        self.complex_graph_allatom_dict = {d.name: d for d in self.complex_graphs_allatom}
        
        # assert len(self.complex_graph_calpha_dict) == len(self.complex_graph_allatom_dict)

        self.full_cache_path = os.path.join(cache_path, f'model_{os.path.splitext(os.path.basename(args.split_train))[0]}'
                                            f'_split_{split}_limit_{limit_complexes}')
        
        # if (not os.path.exists(os.path.join(self.full_cache_path, "ligand_positions.pkl")) and self.cache_creation_id is None) or \
        #         (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
        #     os.makedirs(self.full_cache_path, exist_ok=True)
        #     self.preprocessing()
        
        if (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_rank1_{self.heterographs_name}.pkl"))) and \
           (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_rank1.pkl"))) and \
           (not self.heterographs_combine):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()

        if self.heterographs_name:
            print(f"{self.heterographs_name}_splitting heterographs finish sampling")
            assert False
            
        if self.heterographs_combine:
            all_name, all_ligand_positions, all_ligand_rank1, all_confidences, all_confidence_rank1 = [], {}, {}, {}, {}
            for i in range(5):
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_heterographs_{i}.pkl"), 'rb') as f:
                    names = pickle.load(f)
                    
                with open(os.path.join(self.full_cache_path, f"ligand_positions_heterographs_{i}.pkl"), 'rb') as f:
                    positions_all = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"ligand_positions_rank1_heterographs_{i}.pkl"), 'rb') as f:
                    positions_rank1 = pickle.load(f)
                    
                with open(os.path.join(self.full_cache_path, f"confidences_ligand_positions_heterographs_{i}.pkl"), 'rb') as f:
                    confidences_all = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"confidence_rank1_heterographs_{i}.pkl"), 'rb') as f:
                    confidence_rank1 = pickle.load(f)
                # create a list fomr 1 to 10000
                
                for name in names:
                    all_name.append(name)
                    all_ligand_positions.update(positions_all)
                    all_ligand_rank1.update(positions_rank1)
                    all_confidences.update(confidences_all)
                    all_confidence_rank1.update(confidence_rank1)
            
            assert len(all_name) == len(all_ligand_positions) == len(all_ligand_rank1) == len(all_confidences) == len(all_confidence_rank1)
            print(f'totally {len(all_name)} complexes had been sampled')
            
            with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order.pkl"), 'wb') as f:
                pickle.dump(all_name, f)
            with open(os.path.join(self.full_cache_path, f"ligand_positions.pkl"), 'wb') as f:
                pickle.dump(all_ligand_positions, f)
            with open(os.path.join(self.full_cache_path, f"ligand_positions_rank1.pkl"), 'wb') as f:
                pickle.dump(all_ligand_rank1, f)
            with open(os.path.join(self.full_cache_path, f"confidences_ligand_positions.pkl"), 'wb') as f:
                pickle.dump(all_confidences, f)
            with open(os.path.join(self.full_cache_path, f"confidence_rank1.pkl"), 'wb') as f:
                pickle.dump(all_confidence_rank1, f)
            
            assert False
            
        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading positions and rmsds from: {os.path.join(self.full_cache_path, "ligand_positions_rank1.pkl")}')
            with open(os.path.join(self.full_cache_path, "ligand_positions_rank1.pkl"), 'rb') as f:
                self.full_ligand_positions = pickle.load(f)
                
            if os.path.exists(os.path.join(self.full_cache_path, "confidence_rank1.pkl")):
                with open(os.path.join(self.full_cache_path, "confidence_rank1.pkl"), 'rb') as f:
                    confidences = pickle.load(f)    
                
            if os.path.exists(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl")):
                with open(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"), 'rb') as f:
                    complex_names = pickle.load(f)
            else:
                print('HAPPENING | The path, ', os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"),
                      ' does not exist. \n => We assume that means that we are using a ligand_positions.pkl where the '
                      'code was not saving the complex names for them yet. We now instead use the complex names of '
                      'the dataset that the original model used to create the ligand positions and RMSDs.')
                with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                    original_model_complex_graphs = pickle.load(f)
                    complex_names = [d.name for d in original_model_complex_graphs]
            # assert (len(self.rmsds) == len(complex_names))
        else:
            all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                print(f'HAPPENING | Loading positions and rmsds from cache_id from the path: {os.path.join(self.full_cache_path, "ligand_positions_"+ str(cache_id)+ ".pkl")}')
                if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")): raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
                with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                    full_ligand_positions, rmsds = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                    names_unsorted = pickle.load(f)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)
            names_order = list(set(sum(all_names_unsorted, [])))
            all_rmsds, all_full_ligand_positions, all_names = [], [], []
            for idx, (rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted,all_full_ligand_positions_unsorted, all_names_unsorted)):
                name_to_pos_dict = {name: (rmsd, pos) for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted) }
                intermediate_rmsds = [name_to_pos_dict[name][1] for name in names_order]
                all_rmsds.append((intermediate_rmsds))
                intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
                all_full_ligand_positions.append((intermediate_pos))
            self.full_ligand_positions, self.rmsds = [], []
            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=0))
            for positions_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(positions_tuple, axis=0))
            complex_names = names_order
        print('Number of complex graphs: ', len(complex_names))
        print('Number of RMSDs and positions for the complex graphs: ', len(self.full_ligand_positions))

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))
        assert(len(complex_names) == len(self.full_ligand_positions))
        print(f'the number of {split} dataset: {len(complex_names)}')
        
        if isinstance(self.full_ligand_positions, dict):
            self.positions_dict = {name: self.full_ligand_positions[name] for name in complex_names}
        else:
            self.positions_dict = {name: pos for name, pos in zip(complex_names, self.full_ligand_positions)}
        self.dataset_names = sorted(list(set(self.positions_dict.keys()) & set(self.complex_graph_allatom_dict.keys())))
        # print(self.dataset_names)
        with open(f'dataset_names_{split}', 'wb') as f:
            pickle.dump(self.dataset_names, f)
        # assert False
        
        try:
            if isinstance(confidences, dict):
                self.confidence_dict = {name: confidences[name] for name in complex_names}
            else:
                self.confidence_dict = {name: confidence for name, confidence in zip(complex_names, confidences)}
        except Exception as e:
            print(e)
            pass 
        
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]  
        
    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graph_allatom_dict[self.dataset_names[idx]])
        positions = self.positions_dict[self.dataset_names[idx]]
        assert complex_graph['ligand']['pos'].shape[0] == positions.shape[0]
        confidence = self.confidence_dict[self.dataset_names[idx]]
        complex_graph['confidence'] = confidence

        # if self.balance:
        #     if isinstance(self.rmsd_classification_cutoff, list): raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
        #     label = random.randint(0, 1)
        #     success = rmsds < self.rmsd_classification_cutoff
        #     n_success = np.count_nonzero(success)
        #     if label == 0 and n_success != self.all_samples_per_complex:
        #         # sample negative complex
        #         sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
        #         lig_pos = positions[~success][sample]
        #         complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
        #     else:
        #         # sample positive complex
        #         if n_success > 0: # if no successfull sample returns the matched complex
        #             sample = random.randint(0, n_success - 1)
        #             lig_pos = positions[success][sample]
        #             complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
        #     complex_graph.y = torch.tensor(label).float()
        # else:
        #     sample = random.randint(0, self.all_samples_per_complex - 1)
        #     complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
        #     complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
        #     if isinstance(self.rmsd_classification_cutoff, list):
        #         complex_graph.y_binned = torch.tensor(np.logical_and(rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],rmsds[sample] >= [0] + self.rmsd_classification_cutoff), dtype=torch.float).unsqueeze(0)
        #         complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
        #     complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()
        complex_graph['ligand'].pos = torch.from_numpy(positions)
        complex_graph['ligand'].node_t = {'tr': 0 * torch.ones(complex_graph['ligand'].num_nodes),
                                          'rot': 0 * torch.ones(complex_graph['ligand'].num_nodes),
                                          'tor': 0 * torch.ones(complex_graph['ligand'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': 0 * torch.ones(complex_graph['receptor'].num_nodes),
                                            'rot': 0 * torch.ones(complex_graph['receptor'].num_nodes),
                                            'tor': 0 * torch.ones(complex_graph['receptor'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': 0 * torch.ones(complex_graph['atom'].num_nodes),
                                            'rot': 0 * torch.ones(complex_graph['atom'].num_nodes),
                                            'tor': 0 * torch.ones(complex_graph['atom'].num_nodes)}
        complex_graph.complex_t = {'tr': 0 * torch.ones(1), 'rot': 0 * torch.ones(1), 'tor': 0 * torch.ones(1)}
        return complex_graph.to(self.device)

    def preprocessing(self):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/best_ema_inference_epoch_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        
        if self.confidence_model_dir is not None:
            confidence_model = get_model(self.confidence_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
            state_dict = torch.load(f'{self.confidence_model_dir}/{self.confidence_ckpt}', map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(self.device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None
            confidence_model_args = None
        
        tr_schedule = get_t_schedule(inference_steps=self.inference_steps)
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule
        print('common t schedule', tr_schedule)
        
        #print('HAPPENING | loading cached complexes of the original model to create the confidence dataset RMSDs and predicted positions. Doing that from: ', os.path.join(self.complex_graphs_cache, "heterographs.pkl"))
        #### check if splitting heterographs exists
        # if os.path.exists(os.path.join(self.complex_graphs_cache, f"{self.heterographs_name}.pkl")):
        #     with open(os.path.join(self.complex_graphs_cache, f"{self.heterographs_name}.pkl"), 'rb') as f:
        #         complex_graphs = pickle.load(f)
        # #### 
        # else:
        #     with open(os.path.join(self.complex_graphs_cache, "heterographs.pkl"), 'rb') as f:
        #         complex_graphs = pickle.load(f)
        #print(f'######### in the sampling process, the total number of complexes is {len(complex_graphs)}')
        dataset = ListDataset(self.complex_graphs_calpha)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        full_ligand_positions, confidences_ligand_positions, full_ligand_positions_rank1, confidence_rank1, names = {}, {}, {}, {}, []
        for idx, orig_complex_graph in tqdm(enumerate(loader)):
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max)
            # lig = orig_complex_graph.mol[0]
            if confidence_model is not None and not (self.confidence_model_args.use_original_model_cache or self.confidence_model_args.transfer_weights):
                confidence_data_list = [copy.deepcopy(self.complex_graph_allatom_dict[orig_complex_graph.name[0]]) for _ in range(self.samples_per_complex)]
            else:
                confidence_data_list = None

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None:
                try:
                    predictions_list, confidences = sampling(data_list=data_list, model=model,
                                                             inference_steps=self.inference_steps if self.actual_steps is not None else self.inference_steps,
                                                             tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                                                             device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args, no_random=self.no_random,
                                                             ode=self.ode, visualization_list=None, confidence_model=confidence_model,
                                                             confidence_data_list=confidence_data_list, confidence_model_args=self.confidence_model_args,
                                                             batch_size=self.batch_size, no_final_step_noise=self.no_final_step_noise)
                    
                    ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in predictions_list])
                    if confidences is not None and isinstance(self.confidence_model_args.rmsd_classification_cutoff, list):
                        confidences = confidences[:,0]
                    if confidences is not None:
                        confidences = confidences.cpu().numpy()
                        re_order = np.argsort(confidences)[::-1]
                        confidences = confidences[re_order]
                        ligand_pos = ligand_pos[re_order]
                    # write_dir = f'{args.out_dir}/index{idx}_{data_list[0]["name"][0].replace("/","-")}'
                    # os.makedirs(write_dir, exist_ok=True)
                    # for rank, pos in enumerate(ligand_pos):
                        # mol_pred = copy.deepcopy(lig)
                        # if self.original_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
                        # if rank == 0:
                    full_ligand_positions_rank1[orig_complex_graph.name[0]] = ligand_pos[0]
                    confidence_rank1[orig_complex_graph.name[0]] = confidences[0]
                except Exception as e:
                    if 'failed to converge' in str(e):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                            break
                        print('| WARNING: SVD failed to converge - trying again with a new sample')
                    else:
                        raise e
            if failed_convergence_counter > 5: predictions_list = data_list
            if self.original_model_args.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            # filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

            # if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
            
            # ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
            # orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
            # rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

            # rmsds.append(rmsd)
            # full_ligand_positions.append(np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list]))
            if failed_convergence_counter > 5:
                pass
            else:
                names.append(orig_complex_graph.name[0])
                confidences_ligand_positions[orig_complex_graph.name[0]] = confidences
                full_ligand_positions[orig_complex_graph.name[0]] = ligand_pos
            assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.
        
        with open(os.path.join(self.full_cache_path, f"ligand_positions_rank1{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions_rank1), f)
        with open(os.path.join(self.full_cache_path, f"ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions), f)
            
        with open(os.path.join(self.full_cache_path, f"confidence_rank1{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((confidence_rank1), f)
        with open(os.path.join(self.full_cache_path, f"confidences_ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((confidences_ligand_positions), f)
        
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((names), f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--original_model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained model and hyperparameters')
    parser.add_argument('--restart_dir', type=str, default=None, help='')
    parser.add_argument('--use_original_model_cache', action='store_true', default=False, help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
    parser.add_argument('--data_dir', type=str, default='data/protein_structure_davis', help='Folder containing original structures')
    parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--model_save_frequency', type=int, default=0, help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--best_model_save_frequency', type=int, default=0, help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--run_name', type=str, default='test_confidence', help='')
    parser.add_argument('--project', type=str, default='diffdock_confidence', help='')
    parser.add_argument('--split_train', type=str, default='data/davis_remove_special_protein_train.csv', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/davis_remove_special_protein_val.csv', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')
    parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')
    parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
    parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--gpu_num', type=int, default=0, help='assign the number of gpu')

    # parallel
    parser.add_argument('--heterographs_name', type=str, default=None, help='the name of splitting heterographs')
    parser.add_argument('--heterographs_split_size', type=int, default=None, help='the name of splitting heterographs')
    parser.add_argument('--heterographs_combine', action='store_true', default=False, help='combine id')

    # Inference parameters for creating the positions and rmsds that the confidence predictor will be trained on.
    parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--cache_ids_to_combine', nargs='+', type=str, default=None, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
    parser.add_argument('--cache_creation_id', type=int, default=None, help='number of times that inference is run on the full dataset before concatenating it and coming up with the full confidence dataset')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--inference_steps', type=int, default=2, help='Number of denoising steps')
    parser.add_argument('--samples_per_complex', type=int, default=3, help='')
    parser.add_argument('--balance', action='store_true', default=False, help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
    parser.add_argument('--affinity_prediction', action='store_true', default=True, help='')
    parser.add_argument('--rmsd_classification_cutoff', nargs='+', type=float, default=2, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')

    parser.add_argument('--log_dir', type=str, default='workdir', help='')
    parser.add_argument('--main_metric', type=str, default='confidence_loss', help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
    parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
    parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--w_decay', type=float, default=0.0, help='')
    parser.add_argument('--scheduler', type=str, default='plateau', help='')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='')
    parser.add_argument('--n_epochs', type=int, default=5, help='')

    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='')
    parser.add_argument('--all_atoms', action='store_true', default=True, help='')
    parser.add_argument('--multiplicity', type=int, default=1, help='')
    parser.add_argument('--chain_cutoff', type=float, default=10, help='')
    parser.add_argument('--receptor_radius', type=float, default=30, help='')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='')
    parser.add_argument('--atom_radius', type=float, default=5, help='')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
    parser.add_argument('--matching_popsize', type=int, default=20, help='')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='')
    parser.add_argument('--max_lig_size', type=int, default=50, help='Maximum number of heavy atoms')
    parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='')
    parser.add_argument('--esm_embeddings_path', type=str, default='data/davis_UniProt_ID_embeddings.pt',help='If this is set then the LM embeddings at that path will be used for the receptor features')
    parser.add_argument('--no_torsion', action='store_true', default=False, help='')

    # Model
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
    parser.add_argument('--embedding_scale', type=int, default=10000, help='')
    parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
    parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    common_args = {'batch_size': args.samples_per_complex, 'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir,
                   'confidence_model_dir': args.confidence_model_dir, 'confidence_ckpt': args.confidence_ckpt,
                   'limit_complexes': args.limit_complexes, 'inference_steps': args.inference_steps,
                   'samples_per_complex': args.samples_per_complex, 
                   'all_atoms': args.all_atoms,
                   'no_random': args.no_random, 'ode': args.ode, 'no_final_step_noise': args.no_final_step_noise,
                   'rmsd_classification_cutoff': args.rmsd_classification_cutoff,
                   'use_original_model_cache': args.use_original_model_cache, 'cache_creation_id': args.cache_creation_id, 
                   "cache_ids_to_combine": args.cache_ids_to_combine, "heterographs_name": args.heterographs_name,
                   "heterographs_split_size": args.heterographs_split_size, "heterographs_combine": args.heterographs_combine}
    
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_dataset = AffinityDataset(split="train", device=device, args=args, **common_args)
    

