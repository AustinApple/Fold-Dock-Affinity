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

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
import requests

'''
problems:
1. you do not how to select chain if there are multiple chains in the pdb file.
2. pdb sequence alignment issue, which will cause different residue mapping.
3. you do not know how to select the ligand if there are multiple ligands in the pdb file.
'''

class PDBalign(object):
    def __init__(self, refpdb, samplepdb, ref_chain, sample_chain):
        self.samplepdb = samplepdb
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        self.ref_structure = pdb_parser.get_structure("reference", refpdb)
        self.sample_structure = pdb_parser.get_structure("sample", samplepdb)
        # Use the first model in the pdb-files for alignment 
    
        self.ref_structure = self.ref_structure[0].child_dict[ref_chain]
        self.ref_sequence = self.get_pdb_sequence(self.ref_structure)
        self.sample_structure = self.sample_structure[0].child_dict[sample_chain]
        self.sample_sequence = self.get_pdb_sequence(self.sample_structure)
        self.res_map = self.align_sequence()
        self.ref_atoms = []
        self.sample_atoms = []
        
        # Assertion: Only one chain in the structures!
        # assert len(self.ref_structure.child_dict.keys()) == 1
        # assert len(self.sample_structure.child_dict.keys()) == 1
        # ref_chain_id = self.ref_structure.child_dict.keys()[0]
        # sample_chain_id = self.sample_structure.child_dict.keys()[0]
        # print(self.res_map)
        
        for ref_res in self.res_map:
            try:
                self.ref_atoms.append(self.ref_structure[ref_res]['CA'])
                self.ref_atoms.append(self.ref_structure[ref_res]['N'])
                self.ref_atoms.append(self.ref_structure[ref_res]['O'])
                self.sample_atoms.append(self.sample_structure[self.res_map[ref_res]]['CA'])
                self.sample_atoms.append(self.sample_structure[self.res_map[ref_res]]['N'])
                self.sample_atoms.append(self.sample_structure[self.res_map[ref_res]]['O'])
            except KeyError:
                pass
    
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


def extract_ligands(pdb_id, chain_id = 'A'):
    """ Extraction of the heteroatoms of .pdb files """
    pdb = PDBParser().get_structure("test",f'{pdb_id}/{pdb_id}.pdb')
    io = PDBIO()
    io.set_structure(pdb)
    i = 1
    for model in pdb:
        for chain in model:
            for residue in chain:
                if not is_het(residue):
                    continue
                print(f"saving {chain} {residue}")
                io.save(f"{pdb_id}/ligand_pdb_{i}.pdb", ResidueSelect(chain, residue))
                mol = Chem.MolFromPDBFile(f"{pdb_id}/ligand_pdb_{i}.pdb")
                if (Descriptors.MolWt(mol) > 100) and ((chain.id == chain_id)):
                    io.save(f"{pdb_id}/ligand_pdb.pdb", ResidueSelect(chain, residue))
                i += 1
                
if __name__ == '__main__':
    
    '''
    fetch the pdb file from rcsb and alphafold -> align two structures -> get the rotation matrix and translation matrix ->
    extract the ligand -> align the ligand -> calculate the rmse
    '''
    
    data = pd.read_csv("davis_train_overlap_rcsb_modified.csv", index_col=0)
    data = data[data['pdbid'] == '5MO4']
    with open('ligand_positions_rank1.pkl', 'rb') as f:
        position = pickle.load(f)

    ls_pdbid = list(data['pdbid'])
    ls_uniprot_id = list(data['uniprot_id'])
    ls_name = list(data['name'])
    ls_smi = list(data['ligand_smi'])
    ls_pose_rmse = []
    for pdb_id, uniprot_id, name, smi in zip(ls_pdbid, ls_uniprot_id, ls_name, ls_smi):
        try:
            print(f'################### {pdb_id}')
            pdb_id = pdb_id.lower()
            os.makedirs(pdb_id, exist_ok=True)
            try:
                ## this is for alphafold structure
                URL = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
                response = requests.get(URL)
                open(f'{pdb_id}/{pdb_id}_AF.pdb', "wb").write(response.content)
            except Exception as e:
                print(e)
                pass

            try:
                ## this is for pdb structure
                URL = f'https://files.rcsb.org/download/{pdb_id}.pdb'
                response = requests.get(URL)
                open(f'{pdb_id}/{pdb_id}.pdb', "wb").write(response.content)
            except Exception as e:
                print(e)
                pass

            if os.path.exists(f'{pdb_id}/ligand_pdb.pdb'):
                pass
            else:
                extract_ligands(pdb_id, chain_id='A')

            refpdb = f'{pdb_id}/{pdb_id}.pdb'
            samplepdb = f'{pdb_id}/{pdb_id}_AF.pdb'
            ref_chain = 'A'
            sample_chain = 'A'
            pdbalign = PDBalign(refpdb, samplepdb, ref_chain=ref_chain, sample_chain=sample_chain)
            pdbalign.align()
            pdbalign.write_pdb(outfilename=f'{pdb_id}/{pdb_id}_AF_aligned.pdb')
            rot, tran = pdbalign.rot, pdbalign.tran

            coord = position[str(name)].astype('float')
            new_coord = dot(coord, rot) + tran
            m = Chem.MolFromSmiles(smi)
            # m2=Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            AllChem.MMFFOptimizeMolecule(m)
            conf = m.GetConformer()
            for i in range(conf.GetNumAtoms()):
                x,y,z = new_coord[i]
                conf.SetAtomPosition(i,Point3D(x,y,z))
            print(Chem.MolToMolBlock(m), file=open(f'{pdb_id}/ligand_diff_tran.sdf','w+'))

            mol1 = Chem.MolFromPDBFile(f'{pdb_id}/ligand_pdb.pdb')
            conf = mol1.GetConformer()

            coord_ori = conf.GetPositions()
            assert new_coord.shape == coord_ori.shape
            rmse = np.sqrt(np.mean((new_coord - coord_ori)**2))
            print(rmse)
            ls_pose_rmse.append(rmse)
        except Exception as e:
            print(f'{pdb_id} is problematic')
            print(e)
            print(sys.exc_info()[0])
            traceback.print_exc()



