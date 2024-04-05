import os
import torch


def save_params(output_dir, params, epoch, max_to_keep=None):
    # make sure output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save 
    output_filepath = os.path.join(output_dir, 'itr_{}.pt'.format(epoch))
    torch.save(params, output_filepath)

    # delete files if in excess of max_to_keep
    if max_to_keep is not None:
        files = [os.path.join(output_dir, f) 
                for f in os.listdir(output_dir) 
                if os.path.isfile(os.path.join(output_dir, f)) 
                and 'itr_' in f]
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
        if len(sorted_files) > max_to_keep:
            for filepath in sorted_files[max_to_keep:]:
                os.remove(filepath)
