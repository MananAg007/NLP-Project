

# Import the required library
from tkinter import *
from tkinter import ttk
from demo import *

# Create an instance of tkinter frame
win=Tk()

# Set the geometry
win.geometry("800x450")

def get_input():
    inp0 = text.get(1.0, "end-1c")
    inp1 = text2.get(1.0, "end-1c")

    op = project_demo(inp0, int(inp1))

    label2.config(text= op)

# Add a text widget

# Create a Label widget
label0=Label(win, text="हिंदी क्वेरी यहाँ लिखे", font=('Calibri 12'))
label0.pack()
text=Text(win, width=80, height=5)
text.insert(END, "")
text.pack()


# Create a Label widget
label1=Label(win, text="टेबल क्रमांक?", font=('Calibri 12'))
label1.pack()
text2=Text(win, width=80, height=2)
text2.insert(END, "")
text2.pack()


# Create a button to get the text input
b=ttk.Button(win, text="Print", command=get_input)
b.pack()

# Create a Label widget
label2=Label(win, text="", font=('Calibri 12'))
label2.pack()

win.mainloop()
