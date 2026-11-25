import random


def MLM(text,mask_id,special_tokens_id,vocab):

    original_indices = list(range(len(text)))
    masked_ids= text.copy()

    indices_without_special= []

    for index,i in enumerate(text):
        if i not in special_tokens_id:
            indices_without_special.append(index)


    k= round(len(indices_without_special) * 0.15)

    selected_ids= random.sample(indices_without_special,k)

    first_split_percent= round(len(selected_ids) * 0.80)
    first_split= random.sample(selected_ids,first_split_percent)

    remaining_value= []

    for i in selected_ids:
        if i not in first_split:
            remaining_value.append(i)

    second_split_percent= round(len(selected_ids) * 0.1)
    second_split= random.sample(remaining_value,second_split_percent)
 
    random_id= random.sample(range(vocab),second_split_percent)

    for i in first_split:
        masked_ids[i]= mask_id

    for index_i,i in enumerate(second_split):
           masked_ids[i]= random_id[index_i]

    label_id= [-100] * len(text)

    for i in selected_ids:
        label_id[i]= text[i]

    return masked_ids,label_id
    

if __name__ =="__main__":

    tokenized_text= [12,1342,66,1,33,53,64,100,101,102,103,104,105,106,107,999,0,600,67,99,134,135,500,501,502,503,504,506,566,678,987,567,876,457,346]
    mask_id= [50]
    random_id=[20]

    masked_token_ids= MLM(tokenized_text,mask_id,random_id)

    print(masked_token_ids)

