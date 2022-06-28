import kivy 
from kivy.app import App 
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label 
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock 
from kivy.graphics import Color
from kivy.uix.camera import Camera
from kivy.utils import platform
from kivy.graphics.texture import Texture 
from kivy.properties import ObjectProperty
import time 
import cv2 
import os 

kivy.require('1.9.0')

class MainWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class ThirdWindow(Screen):
    pass

class FourthWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file('ui.kv')

class PuzzleSolverApp(App):
    def build(self):
        return kv

if __name__=="__main__":
    PuzzleSolverApp().run()
