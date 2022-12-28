import os
from skimage import io
import math
import numpy
from info import classes_dict

def psnr(ref, target):
    diff = ref/255.0 - target/255.0
    diff = diff.flatten('C')
    rmse = math.sqrt(numpy.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

test_class = 'ClassC'
class_dir = r'/gpfs/share/home/1801111388/data/ClassC'
ffmpeg = r'/gpfs/share/home/1801111388/ffmpeg/ffmpeg-git-20210425-amd64-static/ffmpeg'
crf_list = [15, 19, 23, 27]
gop_size = 10
enc_frames = 100
if os.path.exists('Report_{}.txt'.format(test_class)):
    os.remove('Report_{}.txt'.format(test_class))

if not os.path.exists(os.path.join(class_dir, 'out')):
    os.makedirs(os.path.join(class_dir, 'out'))

for crf in crf_list:
    for i, yuv_name in enumerate(classes_dict[test_class]["ori_yuv"]):
        bpp_list = []
        intra_bpp_list = []
        psnr_list = []
        w = int(classes_dict[test_class]["resolution"].split('x')[0])
        h = int(classes_dict[test_class]["resolution"].split('x')[1])
        fr = int(classes_dict[test_class]["frameRate"][i])
        short_name = classes_dict[test_class]['sequence_name'][i]
        info = 'FFREPORT=file=ffreport.log:level=56 {bin} -y -pix_fmt yuv420p -s {width}x{height} \
        -r {frame_rate} -i {yuv} -vframes 100 -c:v libx265 -preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={G}:verbose=1" {out_file} ' \
        .format(Q = crf, G = gop_size, out_file = os.path.join(class_dir, 'out', yuv_name.replace('yuv', 'mkv')), 
        bin = ffmpeg, width = w, height = h, frame_rate = fr, yuv = os.path.join(class_dir, 'crop', yuv_name))

        print(info)
        os.system(info)
        if not os.path.exists(os.path.join(class_dir, 'h265', str(crf), short_name)):
            os.makedirs(os.path.join(class_dir, 'h265', str(crf), short_name))
        info = '{bin} -i {out_file} -vframes 100 -f image2 {rec_dir}/im%03d.png' \
            .format(bin = ffmpeg, out_file = os.path.join(class_dir, 'out', yuv_name.replace('yuv', 'mkv')), rec_dir = os.path.join(class_dir, 'h265', str(crf), short_name))

        print(info)
        os.system(info)

        '''
        info = 'python measure265.py {} {} {} > result.txt '.format(os.path.join(class_dir, 'crop', yuv_name), w, h)
        print(info)
        os.system(info)
        '''

        with open('ffreport.log') as f:
            lines = f.readlines()

        for l in lines:
            if "Writing block of" in l:
                bpp_list.append(int(l.split()[7]))

        for i in range(enc_frames):
            if i % gop_size == 0:
                intra_bpp_list.append(bpp_list[i])

            source_path = os.path.join(class_dir, 'images', short_name, 'im{0:0=3d}.png'.format(i + 1))
            h265_path = os.path.join(class_dir, 'h265', str(crf), short_name, 'im{0:0=3d}.png'.format(i + 1))
            source_img = io.imread(source_path)
            h265_img = io.imread(h265_path)
            psnr_val = psnr(source_img, h265_img)
            psnr_list.append(psnr_val)

        print('psnr:', numpy.array(psnr_list).mean())
        print('bpp:', numpy.array(bpp_list).mean() * 8 / (w * h))
        print('intra bpp:', numpy.array(intra_bpp_list).mean() * 8 / (w * h))

        '''
        with open('Report_{}.txt'.format(test_class), 'a') as f:
            f.write('{:.2f} {:.2f} {:.2f}\n'.format(numpy.array(psnr_list).mean(), numpy.array(bpp_list).mean() * 8 / (w * h), numpy.array(intra_bpp_list).mean() * 8 / (w * h)))
        '''
        with open('./intra_bpp/{}_{}_intra_bpp.txt'.format(test_class, crf), 'a') as f:
            f.write('{:.4f}\n'.format(numpy.array(intra_bpp_list).mean() * 8 / (w * h)))
