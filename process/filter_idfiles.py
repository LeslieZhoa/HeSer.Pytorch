import os 
from shutil import rmtree

def rm_little_files(base,th=100):
    k = 1
    for idname in os.listdir(base):
        id_path = os.path.join(base,idname)
        for video_clip in os.listdir(id_path):
            path = os.path.join(id_path,video_clip)
            length = len(os.listdir(path))
            print('\rhave done %04d'%k,end='',flush=True)
            k += 1
            if length < th:
                rmtree(path)
    print()

if __name__ == "__main__":
    base = '../dataset/process'
    rm_little_files(base)