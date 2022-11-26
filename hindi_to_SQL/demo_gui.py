

# Import the required library
from tkinter import *
from tkinter import ttk
from demo import *

# Create an instance of tkinter frame
win=Tk()


# Set the geometry
win.geometry("1200x600")


labelp=Label(win, text="Column Details", font=('Calibri 12'))
labelp.pack()


frame1 = LabelFrame(win, padx=20, pady=20)
frame1.pack()

week_rec_cols = [["Table 0"],["Week", "real", None], ["Date", "string", None], ["Opponent", "string", None], ["Result", "string", None], ["Record", "string", None], ["Game Site", "string", None], ["Attendance", "real", None]]
race_cols = [["Table 1"],["Season", "string", None], ["Series", "string", None], ["Team", "string", None], ["Races", "real", None], ["Wins", "real", None], ["Poles", "real", None], ["F/Laps", "real", None], ["Podiums", "real", None], ["Points", "real", None], ["Position", "string", None]]
party_cols = [["Table 2"],["District", "string", None], ["Incumbent", "string", None], ["Party", "string", None], ["First elected", "real", None], ["Result", "string", None], ["Candidates", "string", None]]
artist_cols = [["Table 3"],["Position", "real", None], ["Artist", "string", None], ["Song title", "string", None], ["Highest position", "real", None], ["Points", "real", None]]

c1 = [i[0] for i in week_rec_cols]
c2 = [i[0] for i in race_cols]
c3 = [i[0] for i in party_cols]
c4 = [i[0] for i in artist_cols]

lst = [c1, c2, c3, c4]


total_rows = len(lst)
total_columns = len(lst[0])



class Table:
    def __init__(self,root):
        # code for creating table
        for i in range(total_rows):
            for j in range(total_columns):
                if j==0:
                    self.e = Entry(root, width= 14, fg='black',
                               font=('Arial', 12, 'bold'))
                else:
                    self.e = Entry(root, width=12, fg='black',
                               font=('Arial',12))
                self.e.grid(row=i, column=j)
                try:
                    self.e.insert(END, lst[i][j])
                except:
                    pass

t= Table(frame1)


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
