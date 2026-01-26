# Batch Writer Skill

**Description**: Enforces strict limits on file writing operations by splitting large content into manageable batches.

---

## ⚠️ CRITICAL RULE - MUST READ ⚠️

**每次使用 Create 或 Edit 工具写入文件时，单次写入内容不得超过 150 行！**

- ✅ 单次写入 ≤ 150 行：允许
- ❌ 单次写入 > 150 行：**禁止**

如果要创建或编辑的内容超过 150 行，**必须分批写入**：
1. 第一批：使用 Create 工具写入前 150 行
2. 后续批次：使用 Edit 工具追加剩余内容，每次最多 150 行

**这是强制性规则，没有例外！**

---

## Purpose

This skill ensures that no single file write operation exceeds 150 lines. When creating or editing files with more than 150 lines, the content must be automatically split into multiple sequential write operations.

## When to Use This Skill

**ALWAYS** use this skill when:
- Creating new files with more than 150 lines of content
- Editing existing files where the replacement content exceeds 150 lines
- Generating large code files, configuration files, or documentation
- Any operation that writes substantial content to a file in a single operation

## Rules

### 1. Line Count Check
Before any file write operation, calculate the total line count of the content to be written:
```python
line_count = len(content.split('\n'))
```

### 2. Batch Splitting Threshold
- **Maximum lines per write**: 150 lines
- If `line_count > 150`, content MUST be split into batches

### 3. Batch Splitting Strategy

When content exceeds 150 lines:

**For New Files (using Create tool):**
```python
# First batch: Write first 150 lines using Create tool
first_batch = '\n'.join(all_lines[:150])
create(file_path, first_batch)

# Subsequent batches: Append remaining lines using Edit tool
for i in range(150, len(all_lines), 150):
    batch = '\n'.join(all_lines[i:i+150])
    # Find appropriate insertion point at end of file
    edit(file_path, old_str=last_lines, new_str=last_lines + '\n' + batch)
```

**For File Editing (using Edit tool):**
- If the new content to insert exceeds 150 lines, split into multiple Edit operations
- Each Edit operation should handle maximum 150 lines of the replacement content
- Use appropriate context markers for each batch

### 4. Smart Boundary Detection

When splitting content:
- **DO NOT** break in the middle of function/class definitions
- **DO NOT** break inside multiline strings or comments
- **DO** break at logical boundaries (after complete functions, classes, or import blocks)
- Maintain proper indentation and syntax validity

### 5. Verification Steps

After completing batch writes:
1. Read the file to verify all content was written correctly
2. Check line count matches expected total
3. Verify syntax validity (for code files)
4. Confirm no duplicate or missing lines

## Implementation Guidelines

### Step 1: Analyze Content
```python
def analyze_content(content):
    lines = content.split('\n')
    total_lines = len(lines)
    batches_needed = (total_lines + 149) // 150  # Ceiling division
    return total_lines, batches_needed
```

### Step 2: Find Smart Split Points
```python
def find_split_point(lines, start_idx, max_lines=150):
    """Find the best place to split content at logical boundaries"""
    end_idx = min(start_idx + max_lines, len(lines))
    
    # Look backward for logical boundaries
    for i in range(end_idx, start_idx + max_lines//2, -1):
        if i >= len(lines):
            continue
        line = lines[i].strip()
        # Split after empty lines, end of functions, end of classes
        if line == '' or line.endswith(':') or line.startswith('#'):
            return i + 1
    
    return end_idx  # Fallback to max_lines
```

### Step 3: Execute Batch Writes
```python
def write_in_batches(file_path, content):
    lines = content.split('\n')
    total_lines = len(lines)
    
    if total_lines <= 150:
        # Single write is safe
        create(file_path, content)
        return
    
    # Multiple batches needed
    current_idx = 0
    batch_num = 1
    
    while current_idx < total_lines:
        split_idx = find_split_point(lines, current_idx)
        batch_content = '\n'.join(lines[current_idx:split_idx])
        
        if batch_num == 1:
            # First batch - create file
            create(file_path, batch_content)
        else:
            # Subsequent batches - append to file
            # Read current file content
            existing_content = read_file(file_path)
            # Append new batch
            edit(file_path, 
                 old_str=existing_content[-100:],  # Last 100 lines as context
                 new_str=existing_content[-100:] + '\n' + batch_content)
        
        current_idx = split_idx
        batch_num += 1
    
    # Verify final result
    final_content = read_file(file_path)
    assert len(final_content.split('\n')) == total_lines, "Line count mismatch!"
```

## Example Usage

### Scenario: Creating a 500-line Python file

**WRONG** (violates 150-line limit):
```python
content = generate_500_lines_of_code()
create('large_file.py', content)  # ❌ Writes 500 lines at once
```

**CORRECT** (respects 150-line limit):
```python
content = generate_500_lines_of_code()
write_in_batches('large_file.py', content)  # ✅ Splits into 4 batches
# Batch 1: Lines 1-150
# Batch 2: Lines 151-300
# Batch 3: Lines 301-450
# Batch 4: Lines 451-500
```

### Scenario: Editing a file with 200-line replacement

**WRONG**:
```python
new_content = generate_200_lines()
edit(file_path, old_str=old_content, new_str=new_content)  # ❌ 200 lines
```

**CORRECT**:
```python
new_content = generate_200_lines()
lines = new_content.split('\n')

# First 150 lines
edit(file_path, 
     old_str=old_content,
     new_str='\n'.join(lines[:150]))

# Remaining 50 lines
edit(file_path,
     old_str=lines[140:150],  # Use last 10 lines as context
     new_str='\n'.join(lines[140:200]))
```

## Error Handling

If any batch write fails:
1. Log the batch number and error details
2. Clean up partial writes if necessary
3. Retry the failed batch with smaller chunk size (e.g., 100 lines)
4. Report completion status to user

## User Feedback

After completing batch writes, provide a summary:
```
✅ File created: example.py
📊 Total lines: 500
📦 Batches written: 4
   - Batch 1: Lines 1-150
   - Batch 2: Lines 151-300
   - Batch 3: Lines 301-450
   - Batch 4: Lines 451-500
✅ All batches verified successfully
```

## Enforcement

This skill is **MANDATORY** for all file operations exceeding 150 lines. The assistant must:
1. Check line count before every write operation
2. Automatically apply batching when needed
3. Provide clear feedback about batching operations
4. Verify content integrity after batch writes

**VIOLATION**: Writing more than 150 lines in a single Create or Edit operation is **STRICTLY PROHIBITED**.
