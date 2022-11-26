

# Import the required library
import tkinter
import tkinter.font as font
from tkinter import *
from tkinter import ttk
from demo import *


# Create an instance of tkinter frame
win=Tk()
win.title("Hindi Text2SQL")

# Set the geometry
win.geometry("1600x900")

def get_input():
    inp0 = text.get(1.0, "end-1c")
    inp1 = text2.get(1.0, "end-1c")

    op = project_demo(inp0, int(inp1))

    label2.config(text= op)

# Add a text widget

# Create a Label widget
label0=Label(win, text="हिंदी क्वेरी यहाँ लिखे", font=('Calibri 20'))
label0.pack()
text=Text(win, width=80, height=6)
text.insert(END, "")
text.pack()
Font_tuple = ("Calibri MS", 20, "bold")
text.configure(font = Font_tuple)


# Create a Label widget
label1=Label(win, text="टेबल क्रमांक?", font=('Calibri 20'))
label1.pack()
text2=Text(win, width=80, height=2)
text2.insert(END, "")
text2.pack()
text2.configure(font = Font_tuple)


# Create a button to get the text input
buttonFont = font.Font(family='Calibri', size=20, weight='bold')
b=tkinter.Button(win, text="Print", command=get_input, font=buttonFont)
b.pack()

# Create a Label widget
label2=Label(win, text="", font=('Calibri 20'))
label2.pack()

win.mainloop()
