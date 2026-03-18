"""Tkinter GUI for generating model-aware prompts."""

import tkinter as tk
from tkinter import ttk, messagebox

from app.engine import (
    MODELS,
    MODEL_DESCRIPTIONS,
    UserInput,
    format_prompt_report,
    generate_prompt_with_report,
)

# ---------------------------------------------------------------------------
# Deep Modern Theme (GitHub Dark Dimmed style)
# ---------------------------------------------------------------------------
_BG_MAIN     = "#0d1117"
_BG_PANEL    = "#161b22"
_BG_INPUT    = "#010409"
_BORDER      = "#30363d"
_FG_PRIMARY  = "#c9d1d9"
_FG_MUTED    = "#8b949e"
_FG_ACCENT   = "#58a6ff"
_FG_GOLD     = "#d29922"

_BTN_GENERATE = "#238636"
_BTN_GENERATE_HOVER = "#2ea043"
_BTN_COPY     = "#1f6feb"
_BTN_COPY_HOVER = "#388bfd"

_FONT_HEADING = ("Segoe UI", 16, "bold")
_FONT_RACE_LABEL = ("Segoe UI", 11, "bold")
_FONT_HINT = ("Segoe UI", 9)
_FONT_INPUT = ("Consolas", 10)
_FONT_BTN = ("Segoe UI", 10, "bold")


class PromptStudioApp:
    def __init__(self, root: tk.Tk):
        self._root = root
        root.title("AI Prompt Generator")
        root.configure(bg=_BG_MAIN)
        root.geometry("1160x900")
        root.minsize(960, 780)

        # Apply basic ttk styling
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        style.configure(
            "TCombobox",
            fieldbackground=_BG_INPUT,
            background=_BORDER,
            foreground=_FG_PRIMARY,
            bordercolor=_BORDER,
            darkcolor=_BG_PANEL,
            lightcolor=_BG_PANEL,
            arrowcolor=_FG_PRIMARY
        )

        # Prompt quality toggles
        self._ask_clarifying = tk.BooleanVar(value=True)
        self._use_planning = tk.BooleanVar(value=False)
        self._use_self_check = tk.BooleanVar(value=False)
        self._auto_enhance = tk.BooleanVar(value=True)
        self._smart_expand = tk.BooleanVar(value=True)
        self._append_report = tk.BooleanVar(value=False)

        self._build_header()
        self._build_body()

    def _build_header(self):
        """Top bar with Title and Engine Selector."""
        header = tk.Frame(self._root, bg=_BG_MAIN)
        header.pack(fill=tk.X, padx=20, pady=(15, 10))

        tk.Label(
            header, text="Prompt Engineering Studio",
            font=_FONT_HEADING, fg=_FG_PRIMARY, bg=_BG_MAIN
        ).pack(side=tk.LEFT)

        # Engine Selector anchored to the right
        right_header = tk.Frame(header, bg=_BG_MAIN)
        right_header.pack(side=tk.RIGHT)

        tk.Label(
            right_header, text="Target Engine:", 
            font=_FONT_RACE_LABEL, fg=_FG_MUTED, bg=_BG_MAIN
        ).pack(side=tk.LEFT, padx=(0, 10))

        self._model_var = tk.StringVar(value=list(MODELS.keys())[0])
        combo = ttk.Combobox(
            right_header, textvariable=self._model_var, 
            values=list(MODELS.keys()), state="readonly", width=25, font=("Segoe UI", 10)
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self._on_engine_change)

        self._desc_var = tk.StringVar()
        self._on_engine_change()

        # Separator line
        tk.Frame(self._root, bg=_BORDER, height=1).pack(fill=tk.X, padx=20, pady=(0, 15))

    def _build_body(self):
        """Split screen: Left=Inputs, Right=Output"""
        body = tk.Frame(self._root, bg=_BG_MAIN)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        body.grid_columnconfigure(0, weight=1, uniform="col")
        body.grid_columnconfigure(1, weight=1, uniform="col")
        body.grid_rowconfigure(0, weight=1)

        # -- LEFT PANEL (Inputs & options) --
        left_panel = tk.Frame(body, bg=_BG_MAIN)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Core input modules
        self._role_txt = self._create_race_module(
            left_panel, "R - Role", "Who should the AI act as? (example: Senior Data Scientist)", 2
        )
        self._action_txt = self._create_race_module(
            left_panel, "A - Action *", "What exact task should it perform?", 4
        )
        self._context_txt = self._create_race_module(
            left_panel, "C - Context", "What background facts, constraints, or data matter?", 5
        )
        self._expect_txt = self._create_race_module(
            left_panel, "E - Expectation", "What output format, style, or constraints are required?", 3
        )
        self._examples_txt = self._create_race_module(
            left_panel,
            "Few-shot Examples",
            "Optional: provide 1-3 input/output examples to steer style and format.",
            5,
        )

        tk.Label(
            left_panel, textvariable=self._desc_var, 
            font=_FONT_HINT, fg=_FG_ACCENT, bg=_BG_MAIN, justify=tk.LEFT, wraplength=450
        ).pack(fill=tk.X, pady=(5, 0))

        # Prompt quality controls
        self._build_quality_panel(left_panel)


        # -- RIGHT PANEL (Outputs & Controls) --
        right_panel = tk.Frame(body, bg=_BG_PANEL, bd=1, relief=tk.SOLID, highlightbackground=_BORDER, highlightcolor=_BORDER, highlightthickness=1)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Output Terminal header
        out_header = tk.Frame(right_panel, bg=_BG_PANEL)
        out_header.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(
            out_header, text="Compiled Prompt Payload", font=_FONT_RACE_LABEL, fg=_FG_PRIMARY, bg=_BG_PANEL
        ).pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="Awaiting Generation...")
        tk.Label(
            out_header, textvariable=self._status_var, font=_FONT_HINT, fg=_FG_MUTED, bg=_BG_PANEL
        ).pack(side=tk.RIGHT)

        # The big output text area
        self._result_txt = tk.Text(
            right_panel, font=_FONT_INPUT, bg=_BG_INPUT, fg=_FG_PRIMARY, 
            insertbackground=_FG_PRIMARY, wrap=tk.WORD, relief=tk.FLAT, padx=15, pady=15
        )
        self._result_txt.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self._result_txt.config(state=tk.DISABLED)

        # Controls at the bottom right
        controls = tk.Frame(right_panel, bg=_BG_PANEL)
        controls.pack(fill=tk.X, padx=15, pady=15)

        tk.Button(
            controls, text="Generate Prompt", bg=_BTN_GENERATE, fg="#ffffff",
            activebackground=_BTN_GENERATE_HOVER, activeforeground="#ffffff",
            font=_FONT_BTN, bd=0, padx=20, pady=8, cursor="hand2", command=self._generate
        ).pack(side=tk.LEFT)

        tk.Button(
            controls, text="Copy", bg=_BTN_COPY, fg="#ffffff",
            activebackground=_BTN_COPY_HOVER, activeforeground="#ffffff",
            font=_FONT_BTN, bd=0, padx=20, pady=8, cursor="hand2", command=self._copy
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            controls, text="Clear All", bg=_BORDER, fg=_FG_PRIMARY, 
            activebackground="#4b535d", activeforeground=_FG_PRIMARY,
            font=("Segoe UI", 9), bd=0, padx=15, pady=8, cursor="hand2", command=self._clear
        ).pack(side=tk.RIGHT)

    def _create_race_module(self, parent, title: str, hint: str, height: int) -> tk.Text:
        """Helper to create a cohesive RACE input block."""
        frame = tk.Frame(parent, bg=_BG_MAIN)
        frame.pack(fill=tk.X, pady=(0, 10))

        lbl_frame = tk.Frame(frame, bg=_BG_MAIN)
        lbl_frame.pack(fill=tk.X)
        
        tk.Label(lbl_frame, text=title, font=_FONT_RACE_LABEL, fg=_FG_PRIMARY, bg=_BG_MAIN).pack(side=tk.LEFT)
        tk.Label(lbl_frame, text=hint, font=_FONT_HINT, fg=_FG_MUTED, bg=_BG_MAIN).pack(side=tk.LEFT, padx=(10, 0))

        txt = tk.Text(
            frame, height=height, font=_FONT_INPUT, bg=_BG_INPUT, fg=_FG_PRIMARY,
            insertbackground=_FG_PRIMARY, wrap=tk.WORD, bd=1, relief=tk.SOLID, 
            highlightthickness=1, highlightbackground=_BORDER, highlightcolor=_FG_ACCENT,
            padx=10, pady=8
        )
        txt.pack(fill=tk.X, pady=(5, 0))
        return txt

    def _build_quality_panel(self, parent: tk.Frame):
        """Builds the prompt quality control panel."""
        panel_container = tk.Frame(parent, bg=_BG_PANEL, bd=1, relief=tk.SOLID, highlightbackground=_FG_GOLD, highlightthickness=1)
        panel_container.pack(fill=tk.X, pady=(15, 0), ipadx=10, ipady=10)

        tk.Label(
            panel_container, text="Prompt Quality Controls",
            font=_FONT_RACE_LABEL, fg=_FG_GOLD, bg=_BG_PANEL
        ).pack(anchor="w", padx=10, pady=(0, 5))

        tk.Label(
            panel_container,
            text=(
                "Safety hardening is always enabled: bypass-style phrases are rewritten "
                "to policy-compliant wording automatically."
            ),
            font=_FONT_HINT,
            fg=_FG_MUTED,
            bg=_BG_PANEL,
            justify=tk.LEFT,
            wraplength=430,
        ).pack(anchor="w", padx=10, pady=(0, 6))

        cb_style = {
            "bg": _BG_PANEL, "fg": _FG_PRIMARY, "activebackground": _BG_PANEL, 
            "activeforeground": _FG_PRIMARY, "selectcolor": _BG_INPUT, "font": _FONT_HINT
        }

        tk.Checkbutton(
            panel_container,
            text="Auto-enhance prompts with generated execution details",
            variable=self._auto_enhance,
            **cb_style
        ).pack(anchor="w", padx=10, pady=2)

        tk.Checkbutton(
            panel_container,
            text="Smart-optimize user prompts into clearer objective/deliverables",
            variable=self._smart_expand,
            **cb_style
        ).pack(anchor="w", padx=10, pady=2)

        tk.Checkbutton(
            panel_container,
            text="Ask clarifying questions first when key details are missing",
            variable=self._ask_clarifying,
            **cb_style
        ).pack(anchor="w", padx=10, pady=2)

        tk.Checkbutton(
            panel_container,
            text="Require a short plan before the final answer",
            variable=self._use_planning,
            **cb_style
        ).pack(anchor="w", padx=10, pady=2)

        tk.Checkbutton(
            panel_container,
            text="Require a self-check against all requirements before final output",
            variable=self._use_self_check,
            **cb_style
        ).pack(anchor="w", padx=10, pady=2)

        tk.Checkbutton(
            panel_container,
            text="Append prompt readiness report to generated payload",
            variable=self._append_report,
            **cb_style
        ).pack(anchor="w", padx=10, pady=2)

    def _on_engine_change(self, event=None):
        model_name = self._model_var.get()
        key = MODELS.get(model_name, "")
        desc = MODEL_DESCRIPTIONS.get(key, "")
        self._desc_var.set(f"Engine Architecture: {desc}")

    def _generate(self):
        action_text = self._action_txt.get("1.0", tk.END).strip()
        if not action_text:
            messagebox.showwarning("Missing Action", "The 'Action' field is the core task and cannot be empty.")
            return

        inp = UserInput(
            role=self._role_txt.get("1.0", tk.END).strip(),
            action=action_text,
            context=self._context_txt.get("1.0", tk.END).strip(),
            expectation=self._expect_txt.get("1.0", tk.END).strip(),
            examples=self._examples_txt.get("1.0", tk.END).strip(),
            ask_clarifying_questions=self._ask_clarifying.get(),
            use_planning=self._use_planning.get(),
            use_self_check=self._use_self_check.get(),
            auto_enhance=self._auto_enhance.get(),
            smart_expand_lazy_input=self._smart_expand.get(),
        )

        model_key = MODELS.get(self._model_var.get(), "chatgpt")
        try:
            res, report = generate_prompt_with_report(model_key, inp)
            if self._append_report.get():
                res = f"{res}\n\n{format_prompt_report(report)}"
            self._result_txt.config(state=tk.NORMAL)
            self._result_txt.delete("1.0", tk.END)
            self._result_txt.insert("1.0", res)
            self._result_txt.config(state=tk.DISABLED)
            
            w_count = len(res.split())
            self._status_var.set(f"✓ Generated • {w_count} words • Quality {report.overall_score}/100")
            self._root.update()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._status_var.set("Error during generation.")

    def _copy(self):
        text = self._result_txt.get("1.0", tk.END).strip()
        if text:
            self._root.clipboard_clear()
            self._root.clipboard_append(text)
            self._root.update()
            self._status_var.set("Copied to clipboard.")
        else:
            messagebox.showinfo("Empty", "Nothing to copy yet.")

    def _clear(self):
        self._role_txt.delete("1.0", tk.END)
        self._action_txt.delete("1.0", tk.END)
        self._context_txt.delete("1.0", tk.END)
        self._expect_txt.delete("1.0", tk.END)
        self._examples_txt.delete("1.0", tk.END)
        self._auto_enhance.set(True)
        self._smart_expand.set(True)
        self._ask_clarifying.set(True)
        self._use_planning.set(False)
        self._use_self_check.set(False)
        self._append_report.set(False)
        
        self._result_txt.config(state=tk.NORMAL)
        self._result_txt.delete("1.0", tk.END)
        self._result_txt.config(state=tk.DISABLED)
        self._status_var.set("Awaiting Generation...")

def main():
    root = tk.Tk()
    PromptStudioApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
