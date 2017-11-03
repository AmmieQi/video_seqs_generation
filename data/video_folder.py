import os
import os.path

VID_EXTENSIONS = [
    '.avi', '.AVI', '.mp4', '.MP4',
]


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)


def make_dataset(dir):
    videos = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
 
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_video_file(fname):
                path = os.path.join(dir, fname)
                videos.append(path)
    return videos

