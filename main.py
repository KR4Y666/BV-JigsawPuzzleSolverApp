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
#import time 
#import cv2 
#import os 

# Define Welcome Screen
class welcome_screen(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
        self.cols = 1                # define number of columns
        self.size_hint = (0.6,0.7)   # define "screen" size as tuple
        self.pos_hint = {
            'center_x': 0.5, 
            'center_y': 0.5
        }  # define screen position as bibliography

        # Add App Name Widget
        name_label = Label(text='Puzzle Solver App',
                           bold=True)
        self.add_widget(name_label)
        
        # Add App Logo Widget
        self.add_widget(Image(source='start.jpeg'))

        # Add Start Button
        self.start_button = Button(text='Start', 
                                   size_hint=(1,0.5),
                                   bold = True,
                                   background_color='d10000',
                                   background_normal='')      
        self.start_button.bind(on_press=self.start_button_func) #bind function to button 
        self.add_widget(self.start_button) 

    # define functionality of start button
    def start_button_func(self, *args):
        Clock.schedule_once(self.switch_to_camera_canvas, 0.5)

    # helper function to switch screens on button 
    def switch_to_camera_canvas(self, *args):
        app.screen_manager.current = 'camera_screen_canvas'

# class for fiest camera screen (canvas)
class camera_screen_canvas(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols = 1

        self.capture_canvas_button = Button(text='Capture Picture of Puzzle Canvas')
        self.capture_canvas_button.bind(on_press = self.capture_canvas_func)
        self.add_widget(self.capture_canvas_button)
        
    # functionality of caputure canvas button
    def capture_canvas_func(self, *args):
        Clock.schedule_once(self.switch_to_camera_piece, 0.5)

    # helper function to move to second camera screen
    def switch_to_camera_piece(self, *args):
        app.screen_manager.current = 'camera_screen_piece'

# class for second camera screen (piece)
class camera_screen_piece(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols = 1

        self.capture_piece_button = Button(text='Capture Picture of Puzzle Piece')
        self.capture_piece_button.bind(on_press=self.capture_piece_func)
        self.add_widget(self.capture_piece_button)

    #define functionality of capture piece button 
    def capture_piece_func(self, *args): 
        Clock.schedule_once(self.switch_to_result_screen, 0.5)

    # helper function to switch to result screen
    def switch_to_result_screen(self, *args): 
        app.screen_manager.current = 'result_screen'

#class for result screen 
class result_screen(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols = 1

        self.result_button = Button(text='Yo hat funktioniert')
        self.add_widget(self.result_button)

# Define Main Class with screen manager
class puzzle_solver(App): 

    def build(self):
        # Add Welcome Screen
        self.screen_manager = ScreenManager()
        self.welcome_screen = welcome_screen()
        screen = Screen(name = 'welcome_screen')
        screen.add_widget(self.welcome_screen)
        self.screen_manager.add_widget(screen)

        # Add Second Screen Aka Camera Screen Canvas
        self.camera_screen_canvas = camera_screen_canvas()
        screen = Screen(name='camera_screen_canvas')
        screen.add_widget(self.camera_screen_canvas)
        self.screen_manager.add_widget(screen)

        # Add third screen aka camera screen piece
        self.camera_screen_piece = camera_screen_piece()
        screen = Screen(name='camera_screen_piece')
        screen.add_widget(self.camera_screen_piece)
        self.screen_manager.add_widget(screen)

        # Add fourth screen aka result screen
        self.result_screen = result_screen()
        screen = Screen(name='result_screen')
        screen.add_widget(self.result_screen)
        self.result_screen.add_widget(screen)

        return self.screen_manager

if __name__ == "__main__":
    
    app = puzzle_solver()
    app.run()