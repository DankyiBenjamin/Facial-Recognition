# main_gui.py

import tkinter as tk
from tkinter import ttk, messagebox
from register_faces import register_user
from recognize_log import recognize_and_log
import pandas as pd
import os

# Ensure necessary folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------------------
# Callback Functions
# ---------------------------


def handle_register():
    user_id = entry_user_id.get().strip()
    if not user_id:
        messagebox.showerror("Error", "Please enter a user ID.")
        return
    register_user(user_id)
    messagebox.showinfo("Success", f"Face registered for: {user_id}")
    entry_user_id.delete(0, tk.END)


def handle_recognize():
    recognize_and_log()
    messagebox.showinfo("Info", "Recognition session ended.")


def show_attendance_log():
    log_path = "logs/attendance.csv"
    if not os.path.exists(log_path):
        messagebox.showinfo("No Logs", "No attendance records found.")
        return

    df = pd.read_csv(log_path)
    log_window = tk.Toplevel(root)
    log_window.title("Attendance Log")

    frame = ttk.Frame(log_window, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    text = tk.Text(frame, wrap=tk.WORD, width=80,
                   height=20, font=("Consolas", 10))
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    if df.empty:
        text.insert(tk.END, "No records found.")
    else:
        text.insert(tk.END, df.to_string(index=False))

    scrollbar = ttk.Scrollbar(frame, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.configure(yscrollcommand=scrollbar.set)


# ---------------------------
# UI Setup
# ---------------------------

root = tk.Tk()
root.title("🎓 Smart Attendance System")
root.geometry("500x350")
root.configure(bg="#f0f0f0")

# Fonts & Colors
heading_font = ("Helvetica", 18, "bold")
label_font = ("Helvetica", 12)
btn_font = ("Helvetica", 11)

# ---------------------------
# Header
# ---------------------------

ttk.Label(root, text="Smart Attendance System",
          font=heading_font, anchor="center").pack(pady=20)

# ---------------------------
# User ID Entry
# ---------------------------

frame_entry = ttk.Frame(root, padding=10)
frame_entry.pack()

ttk.Label(frame_entry, text="Enter User ID:", font=label_font).grid(
    row=0, column=0, padx=5, pady=10)
entry_user_id = ttk.Entry(frame_entry, font=("Helvetica", 11), width=25)
entry_user_id.grid(row=0, column=1, padx=5)

# ---------------------------
# Buttons
# ---------------------------

frame_buttons = ttk.Frame(root, padding=10)
frame_buttons.pack(pady=20)

ttk.Button(frame_buttons, text="📷 Register Face", command=handle_register).grid(
    row=0, column=0, padx=10, ipadx=10, ipady=5)
ttk.Button(frame_buttons, text="🧠 Start Recognition", command=handle_recognize).grid(
    row=0, column=1, padx=10, ipadx=10, ipady=5)
ttk.Button(frame_buttons, text="📄 View Attendance Log", command=show_attendance_log).grid(
    row=1, column=0, columnspan=2, pady=15, ipadx=20, ipady=5)

# ---------------------------
# Footer
# ---------------------------

ttk.Label(root, text="Developed with ❤️ using OpenCV & DeepFace", font=(
    "Helvetica", 9), foreground="#666").pack(side=tk.BOTTOM, pady=10)

# ---------------------------
# Launch the app
# ---------------------------

root.mainloop()
