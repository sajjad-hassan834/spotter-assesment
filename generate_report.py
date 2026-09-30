from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Machine Learning Engineer Assessment Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln(8)

pdf = PDF()
pdf.add_page()
pdf.chapter_title('Validation and Data Split Approach')
body1 = (
    "To validate the model, the 'data/train_test.csv' dataset was split into a "
    "training set and a testing set using an 80/20 split ratio. The split was performed "
    "using a random seed for reproducibility. Since the dataset is large enough, an 80/20 "
    "split provides a sufficient amount of data for the LightGBM model to learn patterns "
    "while holding out a representative sample for evaluation.\n\n"
    "The target variable is 'posted_rate'. Key features were extracted such as 'month', 'day_of_week', "
    "and 'day_of_year' from the 'date' column. Categorical features ('pickup', 'delivery', 'equipment') "
    "were passed directly into LightGBM to utilize its built-in categorical handling capabilities. "
    "Features missing from the December chart inputs (like lat/lon and market signals) were removed "
    "to ensure the model could generalize to the December predictions."
)
pdf.chapter_body(body1)

pdf.chapter_title('December Prediction Chart')
pdf.image('scorer_results/candidate_december.png', x=20, w=170)

pdf.output('report.pdf')
