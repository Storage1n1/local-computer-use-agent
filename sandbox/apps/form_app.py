#!/usr/bin/env python3
import json
import os
import tkinter as tk
from tkinter import messagebox

class RegistrationForm:
    def __init__(self, root):
        self.root = root
        self.root.title("User Registration Form")
        self.root.geometry("520x450")
        self.root.configure(bg="#f0f2f5")

        # Header
        header_frame = tk.Frame(root, bg="#1a73e8", pady=12)
        header_frame.pack(fill=tk.X)
        title_label = tk.Label(header_frame, text="User Registration Form", font=("Helvetica", 16, "bold"), fg="white", bg="#1a73e8")
        title_label.pack()

        # Container
        content = tk.Frame(root, bg="#f0f2f5", padx=30, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        # Full Name
        lbl_name = tk.Label(content, text="Full Name:", font=("Helvetica", 11, "bold"), bg="#f0f2f5", anchor="w")
        lbl_name.pack(fill=tk.X, pady=(5, 2))
        self.entry_name = tk.Entry(content, font=("Helvetica", 12), bg="white", relief=tk.SOLID, bd=1)
        self.entry_name.pack(fill=tk.X, ipady=4, pady=(0, 10))

        # Email
        lbl_email = tk.Label(content, text="Email Address:", font=("Helvetica", 11, "bold"), bg="#f0f2f5", anchor="w")
        lbl_email.pack(fill=tk.X, pady=(5, 2))
        self.entry_email = tk.Entry(content, font=("Helvetica", 12), bg="white", relief=tk.SOLID, bd=1)
        self.entry_email.pack(fill=tk.X, ipady=4, pady=(0, 10))

        # City / Country
        lbl_city = tk.Label(content, text="City / Location:", font=("Helvetica", 11, "bold"), bg="#f0f2f5", anchor="w")
        lbl_city.pack(fill=tk.X, pady=(5, 2))
        self.entry_city = tk.Entry(content, font=("Helvetica", 12), bg="white", relief=tk.SOLID, bd=1)
        self.entry_city.pack(fill=tk.X, ipady=4, pady=(0, 10))

        # Checkbox
        self.agree_var = tk.IntVar(value=0)
        self.chk_agree = tk.Checkbutton(content, text="I agree to the terms and conditions", variable=self.agree_var,
                                        font=("Helvetica", 10), bg="#f0f2f5", activebackground="#f0f2f5")
        self.chk_agree.pack(anchor="w", pady=(5, 15))

        # Submit Button
        btn_frame = tk.Frame(content, bg="#f0f2f5")
        btn_frame.pack(fill=tk.X, pady=10)

        self.btn_submit = tk.Button(btn_frame, text="Submit Form", font=("Helvetica", 12, "bold"),
                                    bg="#1a73e8", fg="white", activebackground="#1558b0", activeforeground="white",
                                    relief=tk.FLAT, padx=20, pady=8, cursor="hand2", command=self.on_submit)
        self.btn_submit.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = tk.Button(btn_frame, text="Clear", font=("Helvetica", 11),
                                   bg="#e0e0e0", fg="#333", activebackground="#d0d0d0",
                                   relief=tk.FLAT, padx=15, pady=8, cursor="hand2", command=self.on_clear)
        self.btn_clear.pack(side=tk.LEFT)

        # Status Banner
        self.lbl_status = tk.Label(content, text="Please enter your details and click Submit",
                                   font=("Helvetica", 11, "italic"), fg="#555555", bg="#f0f2f5", pady=10)
        self.lbl_status.pack(fill=tk.X)

    def on_submit(self):
        name = self.entry_name.get().strip()
        email = self.entry_email.get().strip()
        city = self.entry_city.get().strip()
        agreed = bool(self.agree_var.get())

        if not name:
            self.lbl_status.config(text="Error: Full Name is required!", fg="#d93025")
            return
        if not email:
            self.lbl_status.config(text="Error: Email is required!", fg="#d93025")
            return

        # Save submission
        data = {
            "name": name,
            "email": email,
            "city": city,
            "agreed": agreed,
            "status": "success"
        }
        with open("/tmp/form_submitted.json", "w") as f:
            json.dump(data, f, indent=2)

        self.lbl_status.config(text="Success: Form Submitted Successfully!", fg="#188038", font=("Helvetica", 12, "bold"))
        self.btn_submit.config(state=tk.DISABLED, bg="#34a853")

    def on_clear(self):
        self.entry_name.delete(0, tk.END)
        self.entry_email.delete(0, tk.END)
        self.entry_city.delete(0, tk.END)
        self.agree_var.set(0)
        self.lbl_status.config(text="Form cleared.", fg="#555555")


if __name__ == "__main__":
    root = tk.Tk()
    app = RegistrationForm(root)
    root.mainloop()
