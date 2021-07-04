# import libraries
import numpy as np
import pandas as pd
import seaborn as sn
import cv2
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, flash, request, redirect, render_template, abort
from werkzeug.utils import secure_filename
import os
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# from keras.applications.vgg16 import preprocess_input

# flask constructor
# create an app instance
app = Flask(__name__, template_folder='./template', static_folder='./static')

# flask for colab


# for encrypting the session
app.secret_key = "secret key"
# It will allow below 16MB contents only, you can change it
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# file Upload
path = os.getcwd()

# declared variables
classes = ["C1", "C2", "C3"]
name = []
cnf_matrix = np.empty((3, 3))
cnf_matrix_r = np.empty((3, 3))
did_you_visit_upload = False

# UPLOAD_FOLDER = os.path.join(path, 'uploads')
UPLOAD_FOLDER = './uploads'

# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


# check extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# The route() function of the Flask class is a decorator, which tells the application which URL should call
# the associated function.
@app.route('/')
@app.route('/home')
# ‘/’ and '/home' URL is bound with home function.
def home():
    # render specific template
    # in this case Home.html
    return render_template('Home.html')


@app.route('/about')
def about():
    return render_template('About.html')


@app.route('/upload')
def upload():
    return render_template('Upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    cnt = 0
    i = 1

    print(name)
    # clear names of images from previous session
    if (len(name) != 0):
        name.clear()

    # clear uploads folder for saving new images
    path1 = './uploads/C1'
    path2 = './uploads/C2'
    path3 = './uploads/C3'
    for file_name in os.listdir(path1):
        os.remove(os.path.join(path1, file_name))
    for file_name in os.listdir(path2):
        os.remove(os.path.join(path2, file_name))
    for file_name in os.listdir(path3):
        os.remove(os.path.join(path3, file_name))

    # function to check post request
    if request.method == 'POST' and request.content_length < 16777216:
        # max_size = request.content_length
        # if max_size > 16777216:
        #   return render_template('View.html')
        # check if not empty
        if 'upload_imgs[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        # get files
        files = request.files.getlist('upload_imgs[]')
        print('Number of files uploaded = ', len(files))
        did_you_visit_upload = True

        # save files
        for file in files:
            # check extension
            # file present
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                for i in classes:
                    if (filename.find(i) != -1):
                        name.append(filename)
                        file.save(os.path.join(os.path.join('./uploads/', i), filename))
                        cnt = cnt + 1

                        # save flash message
                        flash(filename + ' successfully uploaded')
                        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    # else:
                    #   flash(filename+' not classified')
            else:
                flash(file.filename + ' has invalid extension and not uploaded')

        print(len(name))

        if (cnt > 0):
            flash('Total ' + str(cnt) + ' File(s) successfully uploaded')
        else:
            flash('No files uploaded')
        return redirect('/upload')
    else:
        print('Done')
        abort(413)
        return render_template('View.html')


def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count


@app.route('/result', methods=['POST', 'GET'])
def result():
    headings = ("Sr. no", "Image name", "ResNet50", "VGG16")
    resnet_prediction, res_accu = resnet()
    vgg_prediction, vgg_accu = vgg()
    r_a = []
    v_a = []
    for i in classes:
        r_a.append(countX(resnet_prediction, i))
        v_a.append(countX(vgg_prediction, i))
    print(r_a)
    print(v_a)
    return render_template('Result.html', headings=headings, resnet=resnet_prediction, vgg=vgg_prediction, name=name,
                           v_a=v_a, r_a=r_a)


@app.route('/analysis')
def analysis():
    path_r = './static/css/images/cfm_r.png'
    path_v = './static/css/images/cfm.png'

    img_r = cv2.imread(path_r)
    cfm_r = cv2.resize(img_r, (510, 510))
    img = cv2.imread(path_v)
    cfm = cv2.resize(img, (510, 510))

    return render_template('Analysis.html', cfm=cfm, cfm_r=cfm_r)


@app.route('/team')
def team():
    return render_template('Team.html')


# testing code for resnet
def resnet():
    # #Prediction of a random test image
    model = load_model('./models/Two_ResNet50_opg_images.h5')
    img_height, img_width = (224, 224)
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       )
    test_generator = train_datagen.flow_from_directory(
        './uploads',
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
    )

    res_loss, res_accu = model.evaluate(test_generator, verbose=2)
    res_accu = "{:.2f}".format((res_accu * 100))
    print(res_accu)

    # prediction conversion from '0,1,2' to 'C1,C2,C3'
    nb_samples = len(test_generator)
    y_prob = []
    y_act = []
    test_generator.reset()

    for _ in range(nb_samples):
        X_test, Y_test = test_generator.next()
        y_prob.append(model.predict(X_test))
        y_act.append(Y_test)

    predicted_resnet = [list(test_generator.class_indices.keys())[i.argmax()] for i in y_prob]
    print(predicted_resnet)
    actual_class = [list(test_generator.class_indices.keys())[i.argmax()] for i in y_act]
    # create confusion matrix
    out_df = pd.DataFrame(np.vstack([predicted_resnet, actual_class]).T, columns=['predicted_class', 'actual_class'])
    confusion_matrix = pd.crosstab(out_df['actual_class'], out_df['predicted_class'], rownames=['actual'],
                                   colnames=['predicted'])

    if os.path.exists('./static/css/images/cfm_r.png'):
        os.remove('./static/css/images/cfm_r.png')
    cfm_plot1 = sn.heatmap(confusion_matrix, cmap='YlGn', annot=True, fmt='d', cbar=False)
    # plt.show()
    cfm_plot1.figure.savefig(os.path.join('./static/css/images', "cfm_r.png"))
    accuracy_print(os.path.join('./static/css/images', "cfm_r.png"), res_accu)
    return (predicted_resnet), res_accu


# testing for vgg16
def vgg():
    model1 = load_model('./models/model_vgg16.h5')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('./uploads',
                                                target_size=(224, 224),
                                                batch_size=1,
                                                class_mode='categorical')

    # prediction conversion from '0,1,2' to 'C1,C2,C3'
    y_prob1 = model1.predict(test_set)
    y_prob1 = np.argmax(y_prob1, axis=1)
    predicted_class1 = y_prob1.tolist()

    print(predicted_class1)
    print(test_set.classes)
    predicted_vgg = []

    cls = ['C1', 'C2', 'C3']
    for i in predicted_class1:
        if (i == 0):
            predicted_vgg.append(cls[0])
        elif (i == 1):
            predicted_vgg.append(cls[1])
        else:
            predicted_vgg.append(cls[2])
    print(predicted_vgg)
    cnf_matrix = confusion_matrix(test_set.classes, predicted_class1)
    print(cnf_matrix)
    path_v = './static/css/images/cfm.png'
    if os.path.exists(path_v):
        os.remove(path_v)

    df_cfm = pd.DataFrame(cnf_matrix, index=classes, columns=classes)
    print(df_cfm)
    plt.figure()

    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d', cbar=False)
    cfm_plot.figure.savefig(os.path.join('./static/css/images', "cfm.png"))
    vgg_accu = "{:.2f}".format((np.diagonal(cnf_matrix).sum() / cnf_matrix.sum().sum() * 100))
    accuracy_print(os.path.join('./static/css/images', "cfm.png"), vgg_accu)


    return predicted_vgg, vgg_accu

def accuracy_print(path, accuracy):
    image = Image.open(path)
    bottom = 100
    width, height = image.size
    new_height = height + bottom
    result = Image.new(image.mode, (width, new_height), (255, 255, 255))
    result.paste(image, (0, 0))
    Acc_vgg = 'Accuracy = ' + str(accuracy) + '%'
    txt = ImageDraw.Draw(result)
    font = ImageFont.truetype('arial.ttf', 32)
    txt.text(((width / 3) - 20, height + 20), Acc_vgg, fill=(0, 0, 0), font=font)
    result.save(path)

# create an app instance
if __name__ == '__main__':
    app.run()
