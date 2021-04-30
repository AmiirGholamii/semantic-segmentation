import os

def revise_fnames(path = 'Development/ScreenshotMasks'):
    print('Revising file names...')
    _, _, filenames = next(os.walk(path))
    for fname in filenames:
        newName = None
        if len(fname) == 9:
            newName = fname[:4] + '000' + fname[4:]
        elif len(fname) == 10:
            newName = fname[:4] + '00' + fname[4:]
        elif len(fname) == 11:
            newName = fname[:4] + '0' + fname[4:]
        else:
            newName = fname
        print(f'Changing {fname} to {newName}')
        os.system(f'mv {os.path.join(path,  fname)} {os.path.join(path, newName)}')


def revise_labels(path = 'Development/ScreenshotMasks'):
    print('Revising file labels...')
    _, _, filenames = next(os.walk(path))
    for fname in filenames:
        outputStr = None
        with open(os.path.join(path, fname), 'r') as f:
            inputStr = f.read()
            outputStr = inputStr
            conversions = {
                '4': ['4', '5', '6', '7', '8', '9', '10', '11', '12'],
                '5': ['13', '14']
            }
            for label in conversions.keys():
                for item in conversions[label]:
                    outputStr = outputStr.replace(item, label)
        with open(os.path.join(path, fname), 'w') as f:
            print(f'writing to {fname}')
            f.write(outputStr)

if __name__ == '__main__':
    revise_fnames()
    revise_labels()
