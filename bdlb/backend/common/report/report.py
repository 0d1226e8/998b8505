from abc import ABC, abstractmethod
import os


class Report:

  def __init__(self):
    self.elements = []
    self.data = {}

  def _add_section(self, title):
    self.elements.append(_Section(title))

  def _add_paragraph(self, text):
    self.elements.append(_Paragraph(text))

  def _add_figure(self,
                  fig,
                  key,
                  caption="",
                  scale=1.0,
                  float=True,
                  centering=True):
    self.elements.append(_Figure(fig, key, caption, scale, float, centering))

  def _add_table(self,
                 table,
                 key,
                 caption=None,
                 centering=True,
                 header=True,
                 index=True):
    self.elements.append(_Table(table, key, caption, centering, header, index))

  def _add_data(self, data, key):
    self.data[key] = data

  def _add_plain_text(self, text):
    self.elements.append(_PlainText(text))

  def to_dict(self):
    """
    Turn report into dict with figures and tables.
    Returns: dict
    """
    d = {}
    for key, value in filter(
        lambda x: x is not None,
        map(lambda x: x.to_key_value_pair(), self.elements)):
      if key in d:
        raise ValueError("Multiple elements with same key")
      else:
        d[key] = value
    return d

  def to_latex(self,
               title,
               date=None,
               author=None,
               a4wide=True,
               include_ambles=True,
               output_file='report.tex'):
    """
    Turn report into .tex file, storing images in same directory
    Args:
        title, str: Title of report
        date, str: Current date
        author, str: Author of report
        a4wide, bool: Whether to use a4wide package
        include_ambles, bool: Whether to include preamble and postamble of tex file
        output_file, str: Path of output file.

    Returns:

    """
    output_file = os.path.abspath(output_file)
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
      os.makedirs(directory)

    result = ""
    if include_ambles:
      result += _create_preamble(title, date=date, author=author, a4wide=a4wide)

    for el in self.elements:
      result += el.to_latex(directory)

    if include_ambles:
      result += "\\end{document}\n"

    with open(output_file, "w") as text_file:
      text_file.write(result)

    return result


class _ReportElement(ABC):

  @abstractmethod
  def to_key_value_pair(self):
    pass

  @abstractmethod
  def to_latex(self, target_folder):
    pass


class _Section(_ReportElement):

  def __init__(self, title):
    self.title = title

  def to_key_value_pair(self):
    return None

  def to_latex(self, target_folder):
    return "\\section{{{}}}\n".format(self.title)


class _PlainText(_ReportElement):

  def __init__(self, text):
    self.text = text

  def to_key_value_pair(self):
    return None

  def to_latex(self, target_folder):
    return self.text + "\n"


class _Paragraph(_ReportElement):

  def __init__(self, text):
    self.text = text

  def to_key_value_pair(self):
    return None

  def to_latex(self, target_folder):
    return "\\paragraph{{{}}}\n".format(self.text)


class _Figure(_ReportElement):

  def __init__(self,
               fig,
               key,
               caption="",
               scale=1.0,
               float=True,
               centering=True):
    self.fig = fig
    self.key = key
    self.scale = scale
    self.float = float
    self.centering = centering
    self.caption = caption

  def to_key_value_pair(self):
    return self.key, self.fig

  def to_latex(self, target_folder, ftype="pdf"):
    self.fig.savefig(
        os.path.join(target_folder, self.key + ".{}".format(ftype)))
    res = "\\begin{figure}"
    if self.float:
      res += "[H]"
    res += "\n"

    if self.centering:
      res += "\\centering\n"

    res += "\\includegraphics[width={}\\linewidth]{{{}}}\n".format(
        self.scale, self.key)

    if len(self.caption) > 0:
      res += "\\caption{{{}}}\n".format(self.caption)

    res += "\\end{figure}\n"

    return res


class _Table(_ReportElement):

  def __init__(self,
               table,
               key,
               caption=None,
               centering=True,
               header=True,
               index=True):
    self.table = table
    self.key = key
    self.header = header
    self.index = index
    self.caption = caption
    self.centering = centering

  def to_key_value_pair(self):
    return self.key, self.table

  def to_latex(self, target_folder):
    result = "\\begin{table}[H]\n"
    if self.centering:
      result += "\\centering\n"
    result += self.table.to_latex(header=self.header, index=self.index)
    if self.caption:
      result += "\\caption{{{}}}\n".format(self.caption)
    result += "\\end{table}\n"
    return result


def _create_preamble(title, date=None, author=None, a4wide=True):
  string = ("\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{float}\n"
            "\\usepackage{graphicx}\n")

  if a4wide:
    string += "\\usepackage{a4wide}\n"

  string += ("\\title{{{}}}\n".format(title))

  if author is not None:
    string += ("\\author{{{}}}\n".format(author))

  if date is not None:
    string += ("\\date{{{}}}\n".format(date))

  string += ("\\begin{document}\n" "\\maketitle\n")

  return string
