import pdfplumber

pdf = pdfplumber.open("Mechatronics by W Bolton.pdf")
page = pdf.pages[0]

print("Words:", len(page.extract_words()))
print("Chars:", len(page.chars))
print("Images:", len(page.images))