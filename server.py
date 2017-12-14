import os
import pickle
import tornado.web
from tornado.ioloop import IOLoop

import numpy as np
from PIL import Image
from io import BytesIO
import cntk as C

# Predict Images
def predict_image(model, image):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    image_width = 224
    image_height = 224

    try:
        img = image
        resized = img.resize((image_width, image_height), Image.ANTIALIAS)
        bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
        hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # compute model output
        arguments = {model.arguments[0]: [hwc_format]}
        output = model.eval(arguments)

        # return softmax probabilities
        sm = C.softmax(output[0])
        return sm.eval()
    except Exception as e:
        print("Could not open the image. {}".format(e))
        return None

model = C.load_model('model/resnet34-inat.model')
id2label = pickle.load(open('metadata/id2label', 'rb'))

class BaseHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        # Get the file
        img = self.request.files.get('file')[0]
        img_body = BytesIO(img['body'])
        # Save the file
        with open(os.path.join('./static/uploads', img['filename']), 'wb') as f:
            image = Image.open(img_body)
            image.thumbnail((299, 299))
            image.save(f, format='JPEG')

        image = Image.open(img_body)
        prediction = predict_image(model, image)
        top_5 = np.argsort(prediction)[-5:]
        pred = []
        for item in reversed(top_5):
            pred.append((id2label[item], prediction[item]))
        print(img['filename'])
        print(pred)

        self.render('predict.html', pred=pred, file=img['filename'])


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/?', BaseHandler),
            (r'/predict/?', UploadHandler)
        ]

        settings = {
            'template_path': os.path.join(os.path.dirname(__file__), 'templates'),
            'static_path': os.path.join(os.path.dirname(__file__), 'static'),
            'debug': True
        }

        super(Application, self).__init__(handlers, **settings)


def main():
    app = Application()
    print('Starting your application at port number 8000')
    app.listen(8000)
    IOLoop.instance().start()


if __name__ == '__main__':
    main()
