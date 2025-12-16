# -*- coding: utf-8 -*-
"""
UI Builder Module

Contains functions to build UI components for the main GUI.
"""

import tkinter as tk
from tkinter import ttk
from .constants import COLORS, BUTTON_STYLE, RADIO_STYLE, LABEL_STYLE_HEADER, TEXT_STYLE_STATS


def create_header(parent):
    """
    Create header frame with title.
    
    Args:
        parent: Parent widget
        
    Returns:
        Header frame
    """
    header_frame = tk.Frame(parent, bg=COLORS['primary'], height=70)
    header_frame.pack(fill=tk.X, padx=0, pady=0)
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(
        header_frame,
        text="NHẬN DIỆN CẢM XÚC THỜI GIAN THỰC",
        font=("Segoe UI", 22, "bold"),
        bg=COLORS['primary'],
        fg=COLORS['text_primary']
    )
    title_label.pack(pady=18)
    
    return header_frame


def create_video_canvas(parent):
    """
    Create video display canvas.
    
    Args:
        parent: Parent widget
        
    Returns:
        Video canvas label
    """
    video_container = tk.Frame(parent, bg=COLORS['bg_dark'], relief=tk.SOLID, bd=1)
    video_container.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 15))
    video_container.grid_rowconfigure(0, weight=1)
    video_container.grid_columnconfigure(0, weight=1)
    
    video_canvas = tk.Label(
        video_container,
        bg='#000000',
        text="Chưa có video\nNhấn 'BẮT ĐẦU QUÉT' để bắt đầu",
        font=("Segoe UI", 12),
        fg='#757575'
    )
    video_canvas.grid(row=0, column=0, sticky='nsew')
    
    return video_canvas


def create_stats_panel(parent):
    """
    Create statistics panel.
    
    Args:
        parent: Parent widget
        
    Returns:
        Stats text widget
    """
    stats_container = tk.Frame(parent, bg=COLORS['bg_dark'], relief=tk.SOLID, bd=1)
    stats_container.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 15))
    stats_container.grid_rowconfigure(0, weight=1)
    stats_container.grid_columnconfigure(0, weight=1)
    
    stats_scrollbar = tk.Scrollbar(stats_container)
    stats_scrollbar.grid(row=0, column=1, sticky='ns')
    
    stats_text = tk.Text(
        stats_container,
        **TEXT_STYLE_STATS,
        state=tk.DISABLED,
        yscrollcommand=stats_scrollbar.set
    )
    stats_text.grid(row=0, column=0, sticky='nsew')
    stats_scrollbar.config(command=stats_text.yview)
    
    return stats_text


def create_control_button(parent, text, command, bg_color, **kwargs):
    """
    Create a styled control button.
    
    Args:
        parent: Parent widget
        text: Button text
        command: Button command
        bg_color: Background color
        **kwargs: Additional button options
        
    Returns:
        Button widget
    """
    button = tk.Button(
        parent,
        text=text,
        bg=bg_color,
        fg=COLORS['text_primary'],
        activebackground=_darken_color(bg_color),
        command=command,
        **{**BUTTON_STYLE, **kwargs}
    )
    return button


def create_radio_button(parent, text, variable, value, command):
    """
    Create a styled radio button.
    
    Args:
        parent: Parent widget
        text: Radio button text
        variable: Tkinter variable
        value: Radio button value
        command: Command to execute on selection
        
    Returns:
        Radio button widget
    """
    radio = tk.Radiobutton(
        parent,
        text=text,
        variable=variable,
        value=value,
        command=command,
        **RADIO_STYLE
    )
    return radio


def create_separator(parent):
    """
    Create a horizontal separator line.
    
    Args:
        parent: Parent widget
        
    Returns:
        Separator frame
    """
    separator = tk.Frame(parent, bg=COLORS['separator'], height=1)
    return separator


def _darken_color(hex_color, factor=0.8):
    """
    Darken a hex color by a factor.
    
    Args:
        hex_color: Hex color string (e.g., '#4CAF50')
        factor: Darkening factor (0-1)
        
    Returns:
        Darkened hex color string
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Darken
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    
    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'


def create_label_with_icon(parent, icon, text, **kwargs):
    """
    Create a label with an icon.
    
    Args:
        parent: Parent widget
        icon: Icon text (emoji or symbol)
        text: Label text
        **kwargs: Additional label options
        
    Returns:
        Label widget
    """
    full_text = f"{icon} {text}"
    label = tk.Label(parent, text=full_text, **kwargs)
    return label
