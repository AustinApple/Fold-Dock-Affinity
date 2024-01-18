import os
import sys
import traceback
import Bio.PDB
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.SCOPData import protein_letters_3to1 as aa3to1
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
from Bio.PDB import PDBParser, PDBIO, Select

import pandas as pd
import pickle
import openbabel
from numpy import array, dot, set_printoptions
import numpy as np
from biopandas.pdb import PandasPdb
from scipy.optimize import minimize, Bounds
from scipy import spatial as spa


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
import requests
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


class PDBalign(object):
    def __init__(self, refpdb, samplepdb, reflig, ref_chain, sample_chain, pocket_cutoff=5):
        self.samplepdb = samplepdb
        self.pocket_cutoff = pocket_cutoff
        
        self.reflig = reflig
        pdbbind_ligand_coords = self.reflig.GetConformer().GetPositions()
        
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        self.ref_structure = pdb_parser.get_structure("reference", refpdb)
        self.sample_structure = pdb_parser.get_structure("sample", samplepdb)
    
    # Use the first model in the pdb-files for alignment 
    
        self.ref_structure = self.ref_structure[0].child_dict[ref_chain]
        self.ref_sequence = self.get_pdb_sequence(self.ref_structure)
        self.sample_structure = self.sample_structure[0].child_dict[sample_chain]
        self.sample_sequence = self.get_pdb_sequence(self.sample_structure)
        
        self.res_map = self.align_sequence()
        # print(self.res_map)
        
        self.ref_alpha_atoms = []
        self.sample_alpha_atoms = []
        
        self.ref_pocket_allatoms = []
        self.ref_pocket_alpha_atoms = []
        self.sample_pocket_allatoms = []
        self.sample_pocket_alpha_atoms = []
        
        
        
        #Assertion: Only one chain in the structures!
        # assert len(self.ref_structure.child_dict.keys()) == 1
        # assert len(self.sample_structure.child_dict.keys()) == 1
        # ref_chain_id = self.ref_structure.child_dict.keys()[0]
        # sample_chain_id = self.sample_structure.child_dict.keys()[0]
        
        
        for ref_res in self.res_map:
            
            try:
                ref_c_alpha_coords = self.ref_structure[ref_res]['CA'].get_coord()
                sample_c_alpha_coords = self.sample_structure[self.res_map[ref_res]]['CA'].get_coord()
                
                ref_res_coords = []
                for atom in self.ref_structure[ref_res].get_atoms():
                    ref_res_coords.append(atom.get_coord())
                    
                sample_res_coords = []
                for atom in self.sample_structure[self.res_map[ref_res]].get_atoms():
                    sample_res_coords.append(atom.get_coord())   
            
                self.ref_alpha_atoms.append(ref_c_alpha_coords)
                self.sample_alpha_atoms.append(sample_c_alpha_coords)
                
                pdbbind_dists = spa.distance.cdist(np.array(ref_res_coords), pdbbind_ligand_coords)
                
                if np.any(pdbbind_dists < self.pocket_cutoff):
                    self.ref_pocket_alpha_atoms.append(ref_c_alpha_coords)
                    self.sample_pocket_alpha_atoms.append(sample_c_alpha_coords)
                    self.ref_pocket_allatoms.append(ref_res_coords)
                    self.sample_pocket_allatoms.append(sample_res_coords)
    
            except KeyError:
                pass
            
        self.ref_alpha_atoms = np.array(self.ref_alpha_atoms)
        self.sample_alpha_atoms = np.array(self.sample_alpha_atoms)
        
        self.ref_pocket_alpha_atoms = np.array(self.ref_pocket_alpha_atoms)
        self.sample_pocket_alpha_atoms = np.array(self.sample_pocket_alpha_atoms)
        self.ref_pocket_allatoms = np.concatenate(self.ref_pocket_allatoms)
        self.sample_pocket_allatoms = np.concatenate(self.sample_pocket_allatoms)
       
            
    def get_pdb_sequence(self, structure):
        """
        Return the sequence of the given structure object
        """
        _aainfo = lambda r: (r.id[1], aa3to1.get(r.resname, 'X'))
        seq = [_aainfo(r) for r in structure.get_residues() if is_aa(r, standard=True)]
        return seq        
            

    def align_sequence(self):
        sample_seq = ''.join([i[1] for i in self.sample_sequence])
        ref_seq = ''.join([i[1] for i in self.ref_sequence])
        # alns = pairwise2.align.globaldx(sample_seq, ref_seq, matlist.blosum62)
        # alns = pairwise2.align.localxx(sample_seq, ref_seq)
        alns = pairwise2.align.globalxx(sample_seq, ref_seq)
        # print(format_alignment(*alns[0]))
        best_aln = alns[0]
        aligned_A, aligned_B, score, begin, end = best_aln      
 
        mapping = {}
        aa_i_A, aa_i_B = 0, 0
        for aln_i, (aa_aln_A, aa_aln_B) in enumerate(zip(aligned_A, aligned_B)):
            if aa_aln_A == '-':
                if aa_aln_B != '-':
                    aa_i_B += 1
            elif aa_aln_B == '-':
                if aa_aln_A != '-':
                    aa_i_A += 1
            else:
                assert self.sample_sequence[aa_i_A][1] == aa_aln_A
                assert self.ref_sequence[aa_i_B][1] == aa_aln_B
                
                if self.ref_sequence[aa_i_B][1] == self.sample_sequence[aa_i_A][1]:
                    mapping[self.ref_sequence[aa_i_B][0]] = self.sample_sequence[aa_i_A][0]
                aa_i_A += 1
                aa_i_B += 1
                
        return mapping

    def align(self):
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(self.ref_atoms, self.sample_atoms)
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        self.sample_structure = pdb_parser.get_structure("sample", self.samplepdb)
        super_imposer.apply(self.sample_structure.get_atoms())
        print(f'protein aligned rmsd: {super_imposer.rms}')
        self.rot, self.tran = super_imposer.rotran

    def write_pdb(self, outfilename=None):
        io = Bio.PDB.PDBIO()
        io.set_structure(self.sample_structure)
        # basename = os.path.splitext(self.samplepdb)[0]
        # if outfilename is None:
        #     outfilename = '%s_aligned.pdb'%basename
        io.save(outfilename)


    def align_prediction(self, smoothing_factor, pdbbind_calpha_coords, alphafold_calpha_coords, pdbbind_ligand_coords, return_rotation=False):
        pdbbind_dists = spa.distance.cdist(pdbbind_calpha_coords, pdbbind_ligand_coords)
        weights = np.exp(-1 * smoothing_factor * np.amin(pdbbind_dists, axis=1))

        pdbbind_calpha_centroid = np.sum(np.expand_dims(weights, axis=1) * pdbbind_calpha_coords, axis=0) / np.sum(weights)
        alphafold_calpha_centroid = np.sum(np.expand_dims(weights, axis=1) * alphafold_calpha_coords, axis=0) / np.sum(weights)
        centered_pdbbind_calpha_coords = pdbbind_calpha_coords - pdbbind_calpha_centroid
        centered_alphafold_calpha_coords = alphafold_calpha_coords - alphafold_calpha_centroid
        centered_pdbbind_ligand_coords = pdbbind_ligand_coords - pdbbind_calpha_centroid

        rotation, rec_weighted_rmsd = spa.transform.Rotation.align_vectors(centered_pdbbind_calpha_coords, centered_alphafold_calpha_coords, weights)
        if return_rotation:
            return rotation, pdbbind_calpha_centroid, alphafold_calpha_centroid

        aligned_alphafold_calpha_coords = rotation.apply(centered_alphafold_calpha_coords)
        aligned_alphafold_pdbbind_dists = spa.distance.cdist(aligned_alphafold_calpha_coords, centered_pdbbind_ligand_coords)
        inv_r_rmse = np.sqrt(np.mean(((1 / pdbbind_dists) - (1 / aligned_alphafold_pdbbind_dists)) ** 2))
        return inv_r_rmse        
        
    
    def get_alignment_rotation(self):
        
        pdbbind_ligand_coords = self.reflig.GetConformer().GetPositions()

        if len(self.ref_alpha_atoms) != len(self.sample_alpha_atoms):
            print('something wrong')
            return None, None, None

        res = minimize(
            self.align_prediction,
            [0.1],
            bounds=Bounds([0.0],[1.0]),
            args=(
                self.ref_alpha_atoms,
                self.sample_alpha_atoms,
                pdbbind_ligand_coords
            ),
            tol=1e-8
        )

        smoothing_factor = res.x
        inv_r_rmse = res.fun
        rotation, pdbbind_calpha_centroid, alphafold_calpha_centroid = self.align_prediction(
            smoothing_factor,
            self.ref_alpha_atoms,
            self.sample_alpha_atoms,
            pdbbind_ligand_coords,
            True
        )

        return rotation, pdbbind_calpha_centroid, alphafold_calpha_centroid
    
    def get_pocket_atoms(self):
        if self.ref_pocket_allatoms.shape == self.sample_pocket_allatoms.shape:
            print(f'using {len(self.ref_pocket_allatoms)} pocket atoms to do the alignment')
            return self.ref_pocket_allatoms, self.sample_pocket_allatoms
        elif self.ref_pocket_alpha_atoms.shape == self.sample_pocket_alpha_atoms.shape:
            print(f'using {len(self.ref_pocket_alpha_atoms)} pocket atoms to do the alignment')
            return self.ref_pocket_alpha_atoms, self.sample_pocket_alpha_atoms
        else:
            print("the number of pocket atoms of alphafold and pdbbind are different")
            

def is_het(residue):
    res = residue.id[0]
    return res != " " and res != "W"

class ResidueSelect(Select):
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def accept_chain(self, chain):
        return chain.id == self.chain.id

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        return residue == self.residue and is_het(residue)
    
    
                
