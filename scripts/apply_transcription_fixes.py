"""
Script để áp dụng fixes vào transcript files.
Có thể chạy standalone hoặc tích hợp vào pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fix_transcription_errors import VietnameseTranscriptionFixer


def fix_transcript_file(input_file: Path, output_file: Path = None, in_place: bool = False):
    """
    Sửa lỗi trong file transcript.
    
    Args:
        input_file: File transcript cần sửa
        output_file: File output (nếu không in-place)
        in_place: Sửa trực tiếp file gốc
    """
    if not input_file.exists():
        print(f"❌ File không tồn tại: {input_file}")
        return False
    
    # Read original content
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    # Fix errors
    fixer = VietnameseTranscriptionFixer()
    fixed_text = fixer.fix_text(original_text)
    
    # Determine output file
    if in_place:
        output_path = input_file
    elif output_file:
        output_path = output_file
    else:
        # Create new file with _fixed suffix
        output_path = input_file.parent / f"{input_file.stem}_fixed{input_file.suffix}"
    
    # Write fixed content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    
    # Show statistics
    if original_text != fixed_text:
        print(f"✅ Đã sửa: {input_file}")
        print(f"   → Output: {output_path}")
        
        # Count changes
        original_words = original_text.split()
        fixed_words = fixed_text.split()
        changes = sum(1 for o, f in zip(original_words, fixed_words) if o != f)
        print(f"   → Số từ đã sửa: {changes}")
    else:
        print(f"ℹ️  Không có lỗi: {input_file}")
    
    return True


def fix_directory(directory: Path, pattern: str = "*.txt", recursive: bool = False):
    """
    Sửa tất cả files trong directory.
    
    Args:
        directory: Directory chứa transcript files
        pattern: Pattern để match files (default: *.txt)
        recursive: Tìm kiếm đệ quy
    """
    if not directory.exists():
        print(f"❌ Directory không tồn tại: {directory}")
        return
    
    # Find files
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    if not files:
        print(f"ℹ️  Không tìm thấy file nào với pattern: {pattern}")
        return
    
    print(f"Tìm thấy {len(files)} file(s)")
    print("=" * 60)
    
    # Process each file
    success_count = 0
    for file in files:
        if fix_transcript_file(file, in_place=False):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Hoàn thành: {success_count}/{len(files)} file(s)")


def interactive_mode():
    """Chế độ interactive để test."""
    fixer = VietnameseTranscriptionFixer()
    
    print("=" * 60)
    print("INTERACTIVE MODE - Vietnamese Transcription Fixer")
    print("=" * 60)
    print("Nhập text để sửa (hoặc 'quit' để thoát)")
    print()
    
    while True:
        try:
            text = input("📝 Nhập text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Tạm biệt!")
                break
            
            if not text:
                continue
            
            fixed = fixer.fix_text(text)
            
            if text != fixed:
                print(f"❌ Sai:  {text}")
                print(f"✅ Đúng: {fixed}")
            else:
                print(f"✅ Text đã đúng!")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Sửa lỗi transcription tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Sửa một file
  python scripts/apply_transcription_fixes.py -f transcripts/transcript.txt
  
  # Sửa tất cả files trong directory
  python scripts/apply_transcription_fixes.py -d transcripts/
  
  # Sửa trực tiếp file gốc (in-place)
  python scripts/apply_transcription_fixes.py -f transcripts/transcript.txt --in-place
  
  # Chế độ interactive
  python scripts/apply_transcription_fixes.py -i
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=Path,
        help='File transcript cần sửa'
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=Path,
        help='Directory chứa transcript files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file (chỉ dùng với -f)'
    )
    
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Sửa trực tiếp file gốc'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        default='*.txt',
        help='Pattern để match files (default: *.txt)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Tìm kiếm đệ quy trong directory'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Chế độ interactive'
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # File mode
    if args.file:
        fix_transcript_file(args.file, args.output, args.in_place)
        return
    
    # Directory mode
    if args.directory:
        fix_directory(args.directory, args.pattern, args.recursive)
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
