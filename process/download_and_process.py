import subprocess
import os 
import pdb 
from multiprocessing import Process
import multiprocessing as mp
import subprocess
import numpy as np

def download_video(q):
    k = 1
    while True:
        vid,save_base = q.get()
        if vid is None:
            break
        cmd = 'yt-dlp -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio \
            https://www.youtube.com/watch?v={vid} \
            --merge-output-format mp4  \
            --output  {save_base}/{vid}.mp4 \
            --external-downloader aria2c \
            --downloader-args aria2c:"-x 16 -k 1M"'.format(vid=vid,save_base=save_base)

        subprocess.call(cmd,shell=True)
        print('\rhave done %06d'%k,end='',flush=True)
        k += 1
    print()
    
def get_frames(q):

    while True:
        path,save_path = q.get()
        if path is None:
            break
        save_base = os.path.split(save_path)[0]
        os.makedirs(save_base,exist_ok=True)
        with open(path,'r') as f:
            lines = f.readlines()
            filter_lines = list(filter(lambda x:x.startswith('0'),lines))
            frames = list(map(lambda x:x.strip().split(),filter_lines))

        np.save(save_path,frames)

def read_file(base,save,q1,q2):
    for idname in os.listdir(base):
        idpath = os.path.join(base,idname)
        for videoname in os.listdir(idpath):
            q1.put([videoname,os.path.join(save,idname,videoname)])
            videopath = os.path.join(idpath,videoname)

            for i,infoname in enumerate(os.listdir(videopath)):
                infopath = os.path.join(videopath,infoname)
                q2.put([infopath,os.path.join(save,idname,videoname,'%02d.npy'%i)])

if __name__ == "__main__":
    base = '../dataset/vox2_test_txt/txt'
    save = '../dataset/voceleb2'
    mp.set_start_method('spawn')
    m = mp.Manager()
    queue1 = m.Queue()
    queue2 = m.Queue()

    read_p = Process(target=read_file,args=(base,save,queue1,queue2,))
    download_p = Process(target=download_video,args=(queue1,))
    frame_p = Process(target=get_frames,args=(queue2,))

    read_p.start()
    download_p.start()
    frame_p.start()

    read_p.join()
    queue1.put([None,None])
    queue2.put([None,None])
    download_p.join()
    frame_p.join()
    