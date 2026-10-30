"""Render every result-dependent document from SAVED results: README, report (+PDF), viva guide, checklist, executed notebook.
Fails with a clear message if results are stale (config fingerprint mismatch) or if template claims no longer hold."""
import re, shutil, subprocess, sys
from . import narrative
from .paths import ROOT


def fill(tpl, F):
    def sub(m):
        k = m.group(1)
        if k not in F: raise KeyError(f"template placeholder {{{{{k}}}}} has no value")
        return str(F[k])
    return re.sub(r"\{\{(\w+)\}\}", sub, tpl)


def render_pdf():
    if not (shutil.which("pandoc")): print("pandoc not found: report.pdf not regenerated (report.md is)"); return False
    try: import weasyprint
    except ImportError: print("weasyprint not installed: report.pdf not regenerated"); return False
    rep = ROOT / "reports"
    subprocess.run(["pandoc", "report.md", "-o", "report.html", "--standalone", "--embed-resources", "--metadata", "pagetitle=Report", "-c", "/dev/null"], cwd=rep, check=True, capture_output=True)
    css = weasyprint.CSS(string="@page{size:A4;margin:16mm} body{font-family:DejaVu Sans,sans-serif;font-size:9.5pt;line-height:1.35;max-width:none} table{border-collapse:collapse;font-size:8pt;width:100%} td,th{border:1px solid #bbb;padding:2px 4px} img{max-width:100%} h1{font-size:15pt;border-bottom:1px solid #999} header h1{font-size:19pt;border:0} code{font-size:8pt;word-break:break-all}")
    weasyprint.HTML(str(rep / "report.html")).write_pdf(str(rep / "report.pdf"), stylesheets=[css]); (rep / "report.html").unlink(); return True


def main(notebook=True):
    F = narrative.derived(narrative.facts()); narrative.guards(F)
    hist = ("# Historical comparison - a DIFFERENT classification task (superseded)\n\n"
            "**Do not compare these numbers with the signal-versus-continuum results.** Before the instructor's clarification, an earlier analysis classified `type == 1` against **all** other codes (0, 2, 3, 4, 5). "
            "Its negative class therefore contained other decays (codes 2-5, 34,841 rows) as well as the 462 continuum rows, the two classes had 35,303 rows each, and the scores say nothing specific about performance against continuum. "
            "The current task keeps only codes 1, 0, -1 (" + F["n_signal"] + " signal vs " + F["n_continuum"] + " continuum). These historical results are not used in any current table, figure or conclusion.\n\n"
            "Full-feature test results of that earlier task (`artifacts/historical_type1_vs_rest/results_full_superseded.csv`):\n\n" + F["table_hist"])
    (ROOT / "reports/historical_type1_vs_all_others.md").write_text(hist); print("rendered reports/historical_type1_vs_all_others.md")
    for tpl, out in (("report.md.tpl", "reports/report.md"), ("README.md.tpl", "README.md"), ("viva_guide.md.tpl", "reports/viva_guide.md"), ("requirements_checklist.md.tpl", "reports/requirements_checklist.md")):
        (ROOT / out).write_text(fill((ROOT / "templates" / tpl).read_text(), F)); print("rendered", out)
    if notebook:
        from . import notebook as nbm
        print("notebook executed, cells:", nbm.build_and_execute(F))
    print("pdf:", render_pdf())


if __name__ == "__main__":
    main()
