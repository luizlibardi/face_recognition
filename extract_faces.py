from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray

detector = MTCNN()

def extract_face(file, size=(160,160)):

    img = Image.open(file) # Caminho completo da foto
    img = img.convert('RGB') # Convertendo em RGB

    array = asarray(img)
    results = detector.detect_faces(array)

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height 

    face = array[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)

    return image

def flip_image(image):
     img = image.transpose(Image.FLIP_LEFT_RIGHT)
     return img


def load_dir(directory_src, directory_target):
    for subdir in listdir(directory_src):
        path = directory_src + subdir + "/"
        path_tgt = directory_target + subdir + "/"

        if not isdir(path):
            continue

        load_pictures(path, path_tgt)
        
def load_pictures(directory_src, directory_target):
    for filename in listdir(directory_src):

        path = directory_src + filename
        path_tgt = directory_target + filename
        path_tgt_flip = directory_target + 'flip-' + filename
        
        try:
            face = extract_face(path)
            flip = flip_image(face)

            face.save(path_tgt, "JPEG", quality=100, optimize=True, progressive=True)
            flip.save(path_tgt_flip, "JPEG", quality=100, optimize=True, progressive=True)
        except:
             print(f'Erro na imagem {format(path)}')
             
if __name__ == '__main__':
        load_dir("/home/luizfernando/Imagens/fotos/", "/home/luizfernando/Imagens/faces/")



