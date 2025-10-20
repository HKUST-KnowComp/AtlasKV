import numpy as np

model_str = "kglam_llama3.1-8b-instruct-10000step-atlas_wiki_qa@test_atlas_cc_qa"
kb_size = 10000
seed = 1607

rough_w_kg = np.load(f'your_result_save_dir/{model_str}__kb_{kb_size}__seed_{seed}_rouge.npy.npz')
rough_wo_kg = np.load(f'your_result_save_dir/{model_str}__kb_{kb_size}__seed_{seed}_rouge_no_kb.npy.npz')

print("*****ROUGH With KG*****")
for key in rough_w_kg.files:
    print(key, rough_w_kg[key])

print("*****ROUGH Without KG*****")
for key in rough_wo_kg.files:
    print(key, rough_wo_kg[key])

acc_w_kg = np.load(f'your_result_save_dir/{model_str}__kb_{kb_size}__seed_{seed}_acc.npy')
conf_w_kg = np.load(f'your_result_save_dir/{model_str}__kb_{kb_size}__seed_{seed}_conf.npy')

print("*****Acc@1 & Acc@5 With KG Across Attn Layers*****")
print(acc_w_kg)
print("*****Conf With KG Across Attn Layers*****")
print(conf_w_kg)