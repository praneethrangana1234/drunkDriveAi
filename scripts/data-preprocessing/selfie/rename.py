# Originally taken from https://github.com/jasonchang0/SoBr/tree/master/bin

from os import rename, listdir, chdir, mkdir
from shutil import move, copy
import random


# need to update for different filetypes
def rename_files():
    try:
        chdir('../../../datasets/frontal-faces/resize_frontal_face')
    except FileExistsError as e:
        print(e)

    count = 0

    for filename in listdir('./'):
        file = filename.split('_')
        name = ""
        filetype = ""
        for part in file:
            # image name/number
            if 'img' in part:
                name = part
            # file extension
            elif '.' in part:
                filetype = part

        # add exception handling 
        if '_UL' in filename:
            rename(filename, '0_' + name + str(count) + filetype)
            count = count + 1

        elif '_LL' in filename:
            rename(filename, '1_' + name + str(count) + filetype)
            count = count + 1

        elif '_UR' in filename:
            rename(filename, '2_' + name + str(count) + filetype)
            count = count + 1

        elif '_LR' in filename:
            rename(filename, '3_' + name + str(count) + filetype)
            count = count + 1

    print(str(count) + ' files renamed!.')


# move files from the new directory to either positive or negative
def move_files():
    try:
        chdir('../../../datasets/frontal-faces/resize_frontal_face')
    except FileExistsError as e:
        print(e)

    for filename in listdir('./'):
        # 0 or 1 glasses of wine => sober
        if filename.startswith('0') or filename.startswith('1'):
            try:
                move('./' + filename, '../negatives/' + filename)
            except FileNotFoundError as e:
                print(e)
        # else 2 or 3 glasses of wine => drunk
        else:
            try:
                move('./' + filename, '../positives/' + filename)
            except FileNotFoundError as e:
                print(e)

    print('Files moved!')


# move a random subset of images. puts extracted samples in new_directory
def extract_random_samples(directory, num_samples):
    chdir('../../../datasets/frontal-faces/')
    try:
        mkdir('selected_' + directory)
    except FileExistsError as e:
        print(e)

    chdir(directory)

    count = 0

    for filename in listdir('./'):
        if random.randint(0, 100) < 50 and count < num_samples:
            print('copy ./' + filename + ' to ' + './../selected_' + directory + '/' + filename)
            try:
                copy('./' + filename, './../selected_' + directory + '/' + filename)
            except IsADirectoryError:
                print('Directory ' + str(directory) + ' does not exist')
                continue
            except FileExistsError:
                print('File ' + filename + ' does not exist')
                continue
            except FileNotFoundError as exception:
                print(exception)
                continue

            count = count + 1

    print(str(count) + ' files were transferred over to new_' + directory)



if __name__ == '__main__':

    # rename_files()
    # move_files()

   # extract_random_samples('negatives', 2500)
    extract_random_samples('positives', 2500)
