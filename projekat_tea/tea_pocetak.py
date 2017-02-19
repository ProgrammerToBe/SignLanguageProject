from matplotlib.pyplot import cm
import itertools
import time
#from main import *
import cv2

from scipy import ndimage
from sympy.physics.secondquant import wicks
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import *
from skimage import color
import numpy as np
#from obrada import *
#from linija import *
import cv2
from sklearn.datasets import fetch_mldata

'''
Odredjivanje linije Probabilistickom Hafovom transformacijom
'''
import math

def mynew_rgb2gray(img_rgb):
    """
    Custom rgb2gray. Forsiramo kanal (boju) koju zelimo da pretvorimo
    u crno
    :param img_rgb:
    :return:
    """
    img_gray = np.ndarray((img_rgb.shape[0],img_rgb.shape[1]))
    img_gray = 1*img_rgb[:, :, 1]
    img_gray = img_gray.astype('uint8')
    frame = img_gray  # img je Numpy array
    # frame = frame[15 : 100, 10 : 100]

    plt.imshow(frame)
    plt.show()

    frame = cv2.resize(frame, (250, 140))

    plt.imshow(frame)
    plt.show()

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    # frame = imutils.resize()
    converted = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # plt.imshow(converted)
    # plt.show()

    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    converted2 = cv2.cvtColor(skin, cv2.COLOR_HSV2RGB)

    # show the skin in the image along with the mask
    plt.imshow(skin)
    plt.show()

    plt.imshow(converted2)
    plt.show()

    img_gray = mynew_rgb2gray(converted2)
    plt.imshow(img_gray, cmap="Greys")
    plt.show()

    print img_gray

    # img_bin = skin > 0

    neki = img_gray > 35

    plt.imshow(neki, cmap="Greys")
    plt.show()

    img_bin = img_gray > 0

    plt.imshow(img_bin, cmap="Greys")
    plt.show()
    img_bin = dilation(img_bin, selem=square(3))
    img_bin = 1 - img_bin

    plt.imshow(img_bin, cmap="Greys")
    plt.show()
    return img_gray


def mynewnew_rgb2gray(img_rgb):
    """
    Custom rgb2gray. Forsiramo kanal (boju) koju zelimo da pretvorimo
    u crno
    :param img_rgb:
    :return:
    """
    img_gray = np.ndarray((img_rgb.shape[0],img_rgb.shape[1]))
    img_gray = 1*img_rgb[:, :, 1]
    img_gray = img_gray.astype('uint8')
    frame = img_gray  # img je Numpy array
    # frame = frame[15 : 100, 10 : 100]

    plt.imshow(frame)
    plt.show()

    frame = cv2.resize(frame, (250, 140))

    plt.imshow(frame)
    plt.show()

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    # frame = imutils.resize()
    converted = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # plt.imshow(converted)
    # plt.show()

    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    converted2 = cv2.cvtColor(skin, cv2.COLOR_HSV2RGB)

    # show the skin in the image along with the mask
    plt.imshow(skin)
    plt.show()

    plt.imshow(converted2)
    plt.show()

    img_gray = mynew_rgb2gray(converted2)
    plt.imshow(img_gray, cmap="Greys")
    plt.show()

    print img_gray

    # img_bin = skin > 0

    neki = img_gray > 35

    plt.imshow(neki, cmap="Greys")
    plt.show()

    img_bin = img_gray > 0

    plt.imshow(img_bin, cmap="Greys")
    plt.show()
    img_bin = dilation(img_bin, selem=square(3))
    img_bin = 1 - img_bin

    img = img_bin

    img[0, :, :] = 1
    img[:, 1, :] = 0
    img[:, :, 2] = 0

    # plt.imshow(img)
    # plt.show()

    img_gray = rgb2gray(img)
    img_tr = img_gray > 0.08
    str_elem = disk(5)
    img_tr_er = erosion(img_tr, selem=str_elem)
    # plt.imshow(img_tr_er, 'gray')
    # plt.show()

    labeled_img = label(img_tr_er)
    regions = regionprops(labeled_img)

    print('Ukupan broj regiona: {}'.format(len(regions)))
    return len(regions)

    plt.imshow(img_bin, cmap="Greys")
    plt.show()
    return img_gray



def racunanje3(img) :
    #img_gray = rgb2gray(img)

    img[0,:,:] = 1
    img[:,1,:] = 0
    img[:,:,2] = 0

    #plt.imshow(img)
    #plt.show()

    img_gray = rgb2gray(img)
    img_tr = img_gray > 0.08
    str_elem = disk(5)
    img_tr_er = erosion(img_tr, selem=str_elem)
    #plt.imshow(img_tr_er, 'gray')
    #plt.show()

    labeled_img = label(img_tr_er)
    regions = regionprops(labeled_img)

    print('Ukupan broj regiona: {}'.format(len(regions)))
    return len(regions)

    ime = 'images/img-'

    #img = imread('images/img-99.png')  # img je Numpy array
    for i in range(0, 100):
        img = img  # img je Numpy array
       # rezultat = racunanje(img)



    with open('out.txt', 'w') as file:
        file.writelines( data )




def tackaULiniju2(pnt, pocetak, kraj):

    line_vec = vector(pocetak, kraj)
    pnt_vec = vector(pocetak, pnt)
    line_lenght = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vector_scaled = scale(pnt_vec, 1.0/line_lenght)
    t = tac(line_unitvec, pnt_vector_scaled)
    r = 1



    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distanca(nearest, pnt_vec)
    nearest = adding(nearest, pocetak)
    return (dist, (int(nearest[0]), int(nearest[1])), r)

def obradjivanje3(ime) :
    img = ime    #imread(ime + str(i) +'.png')  # img je Numpy array

    print img.shape

    for x in range(0, img.shape[0]) :
        for y in range(0, img.shape[1]) :
            if all(v == 0 for v in img[x, y]) :
                img[x, y] = [255, 255, 255]
            else :
                img[x, y] = [0, 0, 0]

    labeled_img = label(img)
    regions = regionprops(labeled_img)

    #plt.imshow(1-regions[0].image)
    #plt.show();

    print len(regions)

    needed = [region for region in regions if region.image.shape == (28L, 28L, 3L)]
    #for region in needed:
     #   print region.image.shape
    ret_val = []
    #plt.imshow(needed[0].image)
    #plt.show()
    for i in range(0, len(needed)):
        min_row = needed[i].bbox[0]
        min_col = needed[i].bbox[1]
        max_row = needed[i].bbox[3]
        max_col = needed[i].bbox[4]

        img_needed = img[min_row:max_row, min_col:max_col]

        #print len(needed)

        img_needed_gray = rgb2gray(img_needed)

        # plt.imshow(255-img_needed_gray*255, cmap="Greys")
        # plt.show()

        ret_val.append(255 - img_needed_gray * 255)

    return ret_val
# http://www.fundza.com/vectors/point2line/index.html
def tac(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distanca(p_prvo, p_drugo):
    return length(vector(p_prvo, p_drugo))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def adding(v, w):
    malo_x, malo_y = v
    veliko_X, veliko_Y = w
    return (malo_x + veliko_X, malo_y + veliko_Y)

def realDeal(frame,edges): #,minLineLength,maxLineGap):

#Probabilisticka Hafova transformacija
#slika, tacnost za udaljenost, tacnost za ugao
#broj glasova neophodan da se uopste smatra linijom
#minimalna duzina linije i maximalni gap
#da bi se dve linije smatrale jednom
    all_found_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30 ,700,10)

    minimum_x=0
    minimum_y=0
    maximum_y=0
    maximum_x=0

    for x0, y0, x1, y1 in all_found_lines[0]:
        minimum_x=x0
        minimum_y=y0
        maximum_x=x1
        maximum_y=y1

#od svih pronadjenih linija trazi se minimalna i maksimalna koordinata
    for i in  range(len(all_found_lines)):
        for x0, y0, x1, y1 in all_found_lines[i]:
            if x0<minimum_x :
                minimum_x=x0
                minimum_y=y0
            if x1>maximum_x:
                maximum_y=y1
                maximum_x=x1

    #cv2.line(frame, (minimum_x,minimum_y), (maximum_x, maximum_y), (0, 255, 0), 2)
    return minimum_x,minimum_y,maximum_x,maximum_y


#Odavde se zove funkcija za pronalazenje ivica i poziva se realDeal
def houghTransformationIntro(frm):#,grayImg):

    gray_image = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

    #Pronalazenje ivica

    #slika, minimalni, maksimalni treshold, velicina Sobelovog kernela
    edges = cv2.Canny(gray_image,100,120,apertureSize = 3)

    minimum_x,minimum_y,maximum_x,maximum_y = realDeal(gray_image,edges)

    cv2.imwrite('drawingOfTheLine.jpg',frm)


    return minimum_x,minimum_y, maximum_x,maximum_y

#Pocetak pronalazenja linije
def findLineParameters(video):
    #ucitavanje videa
    cap = cv2.VideoCapture(video)
    #ucitavanje frejma iz videa
    r, frm = cap.read()

    if not r :
        return #gr = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
   # else:
    #    return


        # print("minx={minx} miny={miny} maxx={maxx} maxy={maxy}".format(minx=minx, miny=miny, maxx=maxx, maxy=maxy))
        #  roughHoughTransformation(frame,edges)
        #  cv2.imwrite('lineDetected13.jpg',frame)

        # print("minx={minx} miny={miny} maxx={maxx} maxy={maxy}".format(minx=minx, miny=miny, maxx=maxx, maxy=maxy))
        #  roughHoughTransformation(frame,edges)

        # cv2.line(frame, (399, 118), (429, 96), (0, 255, 0), 2)
        #  cv2.line(frame, (250, 60), (500, 130), (0, 255, 0), 2)
        # cv2.imwrite('houghlines5.jpg', img)
        # cv2.imwrite('lineDetected13.jpg', frame)
        # return minx, miny, maxx, maxy

        '''
         while(cap.isOpened()):
                    print("Video je ucitan")
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame1=frame

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        break

                print("Prikaz videa")
                print(cv2.__version__)
                cap.release()
                cv2.destroyAllWindows()

        '''

   # print("Prikaz videa")
    print(cv2.__version__)
    cap.release()
    cv2.destroyAllWindows()

    return houghTransformationIntro(frm)#,gr)


#mnist = fetch_mldata('MNIST original', data_home='m_fol')

cc = -1
#dodela ID-jeva
def nextId():
    global cc
    cc += 1
    return cc

#odredjivanje razdaljine trenutnog elementa
# od svih od=stalih koji su pronadjeni
#sa ciljem otkrivanja da li je novi
#ili vec pronadjeni

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distanca(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

#odredjivanje cifre
#na osnovu poredjenja broja razlicitih piksela
def uporediSaMnistom(slika):
    #i=0;
    #while i<70000:
     for i in range(0, 70000):
        suma_razlicitih=0
        mnist_img=new_mnist_set[i]

        suma_razlicitih=np.sum(mnist_img!=slika)


        if suma_razlicitih<15:
            return mnist.target[i]


     return -1

#odredjivanje cifre,
#pretvaranje u sivo

def odrediCifru(img):

    img_bin=(color.rgb2gray(img) >= 0.88).astype('uint8')

    pomerena_slika=putToUpperLeft(img_bin,0)

    rez = uporediSaMnistom(pomerena_slika)
    #print("Procenjeni broj je " + format(rez))
    if rez==-1:
        pomerena_slika_druga_iteracija=putToUpperLeft(img_bin,1)
        rez=uporediSaMnistom(pomerena_slika_druga_iteracija)

    print("Prepoznati broj je ==========================" + format(rez))
    return rez

def draw_regions(regs, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype='float32')
    for reg in regs:
        coords = reg.coords  # coords vraca koordinate svih tacaka regiona
        for coord in coords:
            img_r[coord[0], coord[1]] = 1.
    return img_r

#deo profesorovog koda
def obrada(cap,line):
    suma = 0;
    lista = []
    kernel = np.ones((2,2),np.uint8)

    boundaries = [
        ([230, 230, 230], [255, 255, 255])
    ]


    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 10.0, (640,480))

    elements = []
    t =0
    counter = 0
    times = []


    for rho, theta in lista:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))


    while (1):
        start_time = time.time()
        ret, img = cap.read()
        if not ret:
            break
        (lower, upper) = boundaries[0]
    #obrada slike
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask
        # cv2.erode(img0,kernel)
        #  cv2.erode(img0,kernel3)
        # cv2.erode(img0,kerne32)
        # cv2.erode(img0,kernel2)

        img0 = cv2.dilate(img0, kernel)
        img0 = cv2.dilate(img0, kernel)
        #img0 = cv2.dilate(img0, kernel2)
        #img0 = cv2.dilate(img0, kerne3)
        #img0 = cv2.dilate(img0, kernel3)
        #img0 = cv2.dilate(img0, kernel2)
        #pronalazi objekte koji su pronadjeni na slici i jedinstveno ih oznacava i sadrzani su
        #u objektu labeled, a broj pronadjenih objekata se nalazi u promjenljivoj nr_objects
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)

        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if (dxc > 11 or dyc > 11):

                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                # odredjivanje razdaljine trenutnog elementa
                # od svih od=stalih koji su pronadjeni
                # sa ciljem otkrivanja da li je novi
                # ili vec pronadjeni
                lst = inRange(18, elem, elements)
                nn = len(lst)

                #ukoliko nema onih koji su dovoljno blizu
                #onda je to novi element
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['pass'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []
                    elements.append(elem)
                elif nn == 1:
                    #u suprotnom se apdejtuje stari
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

#za sve pronadjene elemente se odredjuje njihova udaljenost od linije
                #i ukoliko je manja od 10 onda se postavlja flag da je presao liniju
        for el in elements:
            tt = t - el['t']

            if (tt < 10):
                dist, pnt, r = tackaULiniju2(el['center'], line[0], line[1])
                if r > 0:

                    c = (25, 25, 255)
                    if (dist < 15):
                        c = (0, 255, 160)
                        if el['pass'] == False:
                            el['pass'] = True
                            counter += 1
                            (x,y)=el['center']
                            (sx,sy)=el['size']

                            x1=x-14
                            x2=x+14
                            y1=y-14
                            y2=y+14
                            (p1,p2)=(x1,y1)
                            (p3,p4)=(x2,y2)

                            rez = odrediCifru(img[y1:y2,x1:x2])
                            if(rez != -1):
                                suma += rez


                id = el['id']

                for hist in el['history']:
                    ttt = t - hist['t']
                    if (ttt < 100):
                        asd = 1


                for fu in el['future']:
                    ttt = fu[0] - t
                    if (ttt < 100):
                        asd=2


        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Counter: ' + str(counter), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)


        t += 1

       # cv2.imshow('frame', img)
        if (cv2.waitKey(1) & 0xff == ord('q')) :

            break


    out.release()
    cap.release()
    cv2.destroyAllWindows()
    return suma


def calculateItems(miniX,miniY,maxiX,maxiY):
    pminiX = miniX
    pminiY = miniY
    pmaxiX = maxiX
    pmaxiY = maxiY

    mins = (pminiX, pminiY)
    maxs = (pmaxiX, pmaxiY)

    array_of_max = []
    array_of_min = []

    counter1 = 0
    while counter1 <4:
        if(maxs[0]>mins[0]):
            #mins[0]=mins[0]*2
            print("MINS[0] = " + str(mins[0]))
        counter1 +=1

    counter2 = 0
    while counter2 <4:
        if(maxs[1]>mins[1]):
            #mins[1]=mins[1]*2
            print("MINS[1] = " + str(mins[1]))
        counter2 +=1

    #print("trenutno: " + str(mins[1]))

    if mins[1]<100:
        print("trenutno: ")
    #mins[1]= mins[1]+maxs[1]

    if mins[0]<100:
        print("trenutno: ")

    #return mins[0], mins[1], maxs[0], maxs[1]

def pokusaj1():
    im = cv2.imread('images/color_0_0388.png')
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = []
    skin_ycrcb_maxt = []
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    # cv2.imwrite(sys.argv[1], skin_ycrcb) # Second image
    plt.imshow(skin_ycrcb)
    plt.show()

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(im, contours, i, (255, 0, 0), 3)
    # cv2.imwrite(sys.argv[3], im)         # Final image
    plt.imshow(im)
    plt.show()

def pokusaj2():
    im = cv2.imread('images/color_0_0388.png')
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = []
    skin_ycrcb_maxt = []
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    # cv2.imwrite(sys.argv[1], skin_ycrcb) # Second image
    plt.imshow(skin_ycrcb)
    plt.show()

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(im, contours, i, (255, 0, 0), 3)
    # cv2.imwrite(sys.argv[3], im)         # Final image
    plt.imshow(im)
    plt.show()


def cItems2(miniX,miniY,maxiX,maxiY):
    pminiX = miniX
    pminiY = miniY
    pmaxiX = maxiX
    pmaxiY = maxiY

    mins = (pminiX, pminiY)
    maxs = (pmaxiX, pmaxiY)

    array_of_max = []
    array_of_min = []
    im = cv2.imread('images/color_0_0388.png')
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = []
    skin_ycrcb_maxt = []
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    # cv2.imwrite(sys.argv[1], skin_ycrcb) # Second image
    plt.imshow(skin_ycrcb)
    plt.show()

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(im, contours, i, (255, 0, 0), 3)
    # cv2.imwrite(sys.argv[3], im)         # Final image
    plt.imshow(im)
    plt.show()
    counter1 = 0
    while counter1 <4:
        if(maxs[0]>mins[0]):
            #mins[0]=mins[0]*2
            print("MINS[0] = " + str(mins[0]))
        counter1 +=1

    counter2 = 0
    while counter2 <4:
        if(maxs[1]>mins[1]):
            #mins[1]=mins[1]*2
            print("MINS[1] = " + str(mins[1]))
        counter2 +=1
    im = cv2.imread('images/color_0_0388.png')
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = []
    skin_ycrcb_maxt = []
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    # cv2.imwrite(sys.argv[1], skin_ycrcb) # Second image
    plt.imshow(skin_ycrcb)
    plt.show()

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(im, contours, i, (255, 0, 0), 3)
    # cv2.imwrite(sys.argv[3], im)         # Final image
    plt.imshow(im)
    plt.show()
    #print("trenutno: " + str(mins[1]))

    if mins[1]<100:
        print("trenutno: ")
    #mins[1]= mins[1]+maxs[1]

    if mins[0]<100:
        print("trenutno: ")

    #return mins[0], mins[1], maxs[0], maxs[1]

def newItems2(miniX,miniY,maxiX,maxiY):
    pminiX = miniX
    pminiY = miniY
    pmaxiX = maxiX
    pmaxiY = maxiY

    mins = (pminiX, pminiY)
    maxs = (pmaxiX, pmaxiY)

    array_of_max = []
    array_of_min = []
    im = cv2.imread('images/color_0_0388.png')
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = []
    skin_ycrcb_maxt = []
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    # cv2.imwrite(sys.argv[1], skin_ycrcb) # Second image
    plt.imshow(skin_ycrcb)
    plt.show()

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(im, contours, i, (255, 0, 0), 3)
    # cv2.imwrite(sys.argv[3], im)         # Final image
    plt.imshow(im)
    plt.show()
    counter1 = 0
    while counter1 <4:
        if(maxs[0]>mins[0]):
            #mins[0]=mins[0]*2
            print("MINS[0] = " + str(mins[0]))
        counter1 +=1

    counter2 = 0
    while counter2 <4:
        if(maxs[1]>mins[1]):
            #mins[1]=mins[1]*2
            print("MINS[1] = " + str(mins[1]))
        counter2 +=1
    im = cv2.imread('images/color_0_0388.png')
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = []
    skin_ycrcb_maxt = []
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    # cv2.imwrite(sys.argv[1], skin_ycrcb) # Second image
    plt.imshow(skin_ycrcb)
    plt.show()

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(im, contours, i, (255, 0, 0), 3)
    # cv2.imwrite(sys.argv[3], im)         # Final image
    plt.imshow(im)
    plt.show()
    #print("trenutno: " + str(mins[1]))



def calculateItems22(miniX,miniY,maxiX,maxiY):
    pminiX = miniX
    pminiY = miniY
    pmaxiX = maxiX
    pmaxiY = maxiY

    mins = (pminiX, pminiY)
    maxs = (pmaxiX, pmaxiY)

    array_of_max = []
    array_of_min = []

    counter1 = 0
    while counter1 <4:
        if(maxs[0]>mins[0]):
            #mins[0]=mins[0]*2
            print("MINS[0] = " + str(mins[0]))
        counter1 +=1

    counter2 = 0
    while counter2 <4:
        if(maxs[1]>mins[1]):
            #mins[1]=mins[1]*2
            print("MINS[1] = " + str(mins[1]))
        counter2 +=1

    #print("trenutno: " + str(mins[1]))

    if mins[1]<100:
        print("trenutno: 1")
    #mins[1]= mins[1]+maxs[1]

    if mins[0]<100:
        print("trenutno: 0")

    #return mins[0], mins[1], maxs[0], maxs[1]

def my_rgb2gray2(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.5*img_rgb[:, :, 0] + 0.0*img_rgb[:, :, 1] + 0.5*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray





#Nalazenje granica regiona od znacaja
#odredjivanjem rastojanja svaka dva regiona
#odredjivanjem srednje vrednosti tih rastojanja
#odredjivanjem srednje vrednosti tih srednjih rastojanja
#i uzimanjem u obzir samo onih
# cije je srednje rastojanje manje od 1/3
# globalnog  srednjeg rastojanja

def findBoundariesOfRegions(img_bin):
    label_img = label(img_bin)
    regions = regionprops(label_img)

    regions_objects=[]

    for region in regions:

        regXX={'reg_bbox': region.bbox, 'center': (round((region.bbox[0]+region.bbox[2])/2),
                                                 round((region.bbox[1]+region.bbox[3])/2)), 'status': 'right'}
        regions_objects.append(regXX)

    regioni_n = []
    global_distance=0
    n=len(regions_objects)

    if n>1:
    #ukoliko je vec samo jedan region
    #ne moze se nista popraviti ovom metodom
        a = 0
        b = 0
        c = 0
        d = 0
        for x1, y1, x2, y2 in regioni_n:
            a = x1
            b = y1
            c = x2
            d = y2

        if n==2:
            # ukoliko su u pitanju samo dva regiona
            # od znacaja je samo onaj
            # koji ima vise piksela
            if(regions[0].area > regions[1].area) :

                return regions[0].bbox[0],regions[0].bbox[2],regions[0].bbox[1], regions[0].bbox[3]
            else :
                return regions[1].bbox[0], regions[1].bbox[2], regions[1].bbox[1], regions[1].bbox[3]

        else:

            # Nalazenje granica regiona od znacaja
            # odredjivanjem rastojanja svaka dva regiona
            # odredjivanjem srednje vrednosti tih rastojanja
            # odredjivanjem srednje vrednosti tih srednjih rastojanja
            # i uzimanjem u obzir samo onih
            # cije je srednje rastojanje manje od 1/3
            # globalnog  srednjeg rastojanja


            for reg_i in regions_objects:
                dist=0
                for reg_j in regions_objects:
                    if(reg_i['center']!=reg_j['center']):
                        dist+=pow((reg_i['center'][0]-reg_j['center'][0]),2)+pow((reg_i['center'][1]-reg_j['center'][1]),2)
                reg_i['dist']=dist/(n-1)
                global_distance+=dist/(n-1)

            average_global_distance=1.3*global_distance/n

            for reg in regions_objects:

                if reg['dist']>average_global_distance:

                    reg['status']='chosen'
                else:
                    reg['status']='not_chosen'


    minimum_x=100
    maximum_x=-100
    minimum_y=100
    maximum_y=-100
    for reg in regions_objects:
        if(reg['status']=='chosen'):
            bbox=reg['reg_bbox']
            if bbox[0] < minimum_x:
                minimum_x = bbox[0]
            if bbox[1] < minimum_y:
                minimum_y = bbox[1]
            if bbox[2] > maximum_x:
                maximum_x = bbox[2]
            if bbox[3] > maximum_y:
                maximum_y = bbox[3]

    #calculateItems(minimum_x,minimum_y,maximum_x,maximum_y)
    return minimum_x,maximum_x,minimum_y,maximum_y

#smestanje u gornji levi ugao
#pronalazenjem najmanjeg pravougaonika
#koji obuhvata sve regione
#
def putToUpperLeft(img_bin,n):

    labele_slike = label(img_bin)
    regions = regionprops(labele_slike)

    minimum_x=800
    minimum_y=800
    maximum_x=-1
    maximum_y=-1

    if(n==1):
        #ovo se izvrsi za slucaj da je doslo do toga
        #da nije mogla da se odredi cifra u prvoj iteraciji
        # pa, ukoliko je doslo do situacije u kojoj su cifre preblizu
        # i jedan deo regiona jedne je upao u odsecak trenutno obradjivane
        # onda je neophodno izolovati regione od vaznosti
        minimum_x,maximum_x,minimum_y,maximum_y=findBoundariesOfRegions(img_bin)
    else:
        #ukoliko je ovo obrradjivanje mnista
        #ili prva iteracija prepoznavanja cifre
        #iyvrsava se ovaj deo
        for reg in regions:
            reg_bbox = reg.bbox
            if reg_bbox[0]<minimum_x:
                minimum_x=reg_bbox[0]
            if reg_bbox[1] <minimum_y:
                minimum_y=reg_bbox[1]
            if reg_bbox[2]>maximum_x:
                maximum_x=reg_bbox[2]
            if reg_bbox[3]>maximum_y:
                maximum_y=reg_bbox[3]


    height = maximum_x - minimum_x
    width = maximum_y - minimum_y

    imgForReturn = np.zeros((28, 28))

    imgForReturn[0:height, 0:width] = imgForReturn[0:height, 0:width] + img_bin[minimum_x:maximum_x, minimum_y:maximum_y]


    return imgForReturn

new_mnist_set=[]
#smestanje svih cifara iz mnista u gornji levi ugao
#pravljennje novog mnist seta
def preradiMnist(mnist):
    for i in range(0, 70000):
        mnist_img=mnist.data[i].reshape(28,28)
        mnist_img_bin=((color.rgb2gray(mnist_img)/255.0)>0.88).astype('uint8')

        new_mnist_img=putToUpperLeft(mnist_img_bin,0)
        new_mnist_set.append(new_mnist_img)

#ucitavanje originalnog mnista
mnist = fetch_mldata('MNIST original')
preradiMnist(mnist)

with open('out.txt', 'r') as file:
    # read a list of lines into data
    dataLines = file.readlines()


#obrada svih videa
for i in range(0,10):
    videoName = "Sestica/mvideo-" + format(i) + ".avi"
    ime = 'video-' + format(i)+ '.avi'
    cap = cv2.VideoCapture(videoName)

    x1, y1, x2, y2 = findLineParameters(videoName)


    linija = [(x1, y1), (x2, y2)]
    print("Video- "+format(i))
    suma = obrada(cap,linija)

    delovi = dataLines[2 + i].split('\t')
    dataLines[2 + i] = delovi[0] + '\t' + str(suma) + "\n"

    print suma

with open('out.txt', 'w') as file:
    file.writelines(dataLines)


def upis() :


    with open('out.txt', 'r') as file:
    # read a list of lines into data
        dataLines = file.readlines()

    ime = 'images/img-'

    mina = 1;
    minb = 1;
    minc = -1;
    mind = -1;
    #for w in range(0, 100):
    #img_name = ime + str(w) + '.png'  # img je Numpy array
    slike = []

    # print len(slike)
    suma = 0;
    for i in range(0, len(slike)):
        img_neka = slike[i]
        test = img_neka.reshape(1, -1)

    w = suma/2;
    delovi = dataLines[2 + w].split('\t')
    #dataLines[2 + w] = delovi[0] + '\t' + str(suma) + "\n"

    with open('out.txt', 'w') as file:
        file.writelines(dataLines)


'''
file = open("out.txt", "r+")
data = file.read()
lines = data.split('\n')
print lines

with open('out.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()
'''
def rgb2gray(img):

    return img>0.8

def racunanje(img) :
    #img_gray = rgb2gray(img)

    img[0,:,:] = 1
    img[:,1,:] = 0
    img[:,:,2] = 0

    #plt.imshow(img)
    #plt.show()

    img_gray = rgb2gray(img)
    img_tr = img_gray > 0.08
    str_elem = disk(5)
    img_tr_er = erosion(img_tr, selem=str_elem)
    #plt.imshow(img_tr_er, 'gray')
    #plt.show()

    labeled_img = label(img_tr_er)
    regions = regionprops(labeled_img)

    print('Ukupan broj regiona: {}'.format(len(regions)))
    return len(regions)

ime = 'images/img-'

#img = imread('images/img-99.png')  # img je Numpy array
#for i in range(0, 100):
