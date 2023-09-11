import requests
import json
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

def get_PDBid_from_uniprot(uniprot=None):
    '''get all the PDBids from a uniprot id'''
    payload = {
      "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "operator": "exact_match",
              "value": uniprot,
              "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession"
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "operator": "exact_match",
              "value": "UniProt",
              "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name"
            }
          }
        ]
      },
      "request_options": {
        "return_all_hits": True
      },
      "return_type": "entry"
    }


    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    response = requests.post(url, json=payload)
    response_json = response.json()
    return [i['identifier'] for i in response_json['result_set']]

def get_smi_from_pdbid(pdbid=None):
    '''get all the molecular SMILES from a PDBid'''
    ls_smi = []
    try:
        query = f'''{{
          entry(entry_id: "{pdbid}") {{
            nonpolymer_entities {{
              nonpolymer_comp {{
                rcsb_id
                pdbx_chem_comp_descriptor {{
                  type
                  comp_id
                  program
                  descriptor
                }}
              }}
            }}
          }}
        }}'''
        url = "https://data.rcsb.org/graphql"
        r = requests.post(url, json={'query': query})
        for i in r.json()['data']['entry']['nonpolymer_entities']:
            if i['nonpolymer_comp']['pdbx_chem_comp_descriptor'][0]['type'] == 'SMILES':
                ls_smi.append(Chem.CanonSmiles(i['nonpolymer_comp']['pdbx_chem_comp_descriptor'][0]['descriptor']))
        return ls_smi
    except TypeError:
        return []

if __name__ == '__main__':
    davis = pd.read_csv("DiffDock/data/davis_remove_special_protein.csv")
    ls_davis_uniprot = list(set(davis['uniprot_id']))
    
    dict_uniprot_smi_davis = {}
    for uniprot in davis.groupby("uniprot_id"):
        dict_uniprot_smi_davis[uniprot[0]] = set(Chem.CanonSmiles(smi) for smi in list(uniprot[1]['ligand_smi']))
        
    dict_uniprot_smi_rcsb = {}
    dict_uniprotsmi_pdbid = {}
    for uniprot in ls_davis_uniprot:
        ls_smi_all = []
        try:
            for pdbid in get_PDBid_from_uniprot(uniprot):
                ls_smi = get_smi_from_pdbid(pdbid)
                if ls_smi:
                    for smi in ls_smi:
                        dict_uniprotsmi_pdbid[f'{uniprot}_{smi}'] = pdbid
                ls_smi_all.extend(ls_smi)
            dict_uniprot_smi_rcsb[uniprot] = set(ls_smi_all)
        except Exception as e:
            dict_uniprot_smi_rcsb[uniprot] = None
            
    overlap_pair = [] # overlap pair of (uniprot, smi)

    for uniprot in dict_uniprot_smi_davis:
        try:
            overlap = dict_uniprot_smi_davis[uniprot] & dict_uniprot_smi_rcsb[uniprot]
            if overlap:
                for smi in overlap:
                    overlap_pair.append((uniprot, smi))
        except Exception as e:
            pass
        
    data_overlap = pd.DataFrame(columns=['uniprot', 'smiles', 'pdbid'])
    data_overlap['uniprot'] = [pair[0] for pair in overlap_pair]
    data_overlap['smiles'] = [pair[1] for pair in overlap_pair]
    data_overlap['pdbid'] = [dict_uniprotsmi_pdbid[f'{pair[0]}_{pair[1]}'] for pair in overlap_pair]
    # there should be 102 complexes in davis dataset that overlapped wit rcsb (having pdbid)
    data_overlap.to_csv("davis_overlap_rcsb.csv", index=False)







