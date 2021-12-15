import unittest
import app

BASE_URL = 'http://127.0.0.1:5000/'



class TestFlaskApi(unittest.TestCase):

    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    def test_svm(self):
        rv = self.app.get(BASE_URL+'uploader')
        self.assertEqual(rv.status, '200 OK')

    def test_vgg(self):

        rv = self.app.get(BASE_URL+'uploadervgg')
        self.assertEqual(rv.status, '200 OK')
        

if __name__ == '__main__':
    import xmlrunner
    runner = xmlrunner.XMLTestRunner(output='test-reports')
    unittest.main(testRunner=runner)