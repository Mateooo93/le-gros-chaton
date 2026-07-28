[![ACL Logo](https://aclanthology.org/images/acl-logo.svg)ACL Anthology](https://aclanthology.org/)

- [About](https://aclanthology.org/2025.findings-acl.428/#)
  - [Announcements](https://aclanthology.org/posts/)
  - [Communication channels](https://aclanthology.org/faq/news/)
  - [Related work](https://aclanthology.org/faq/related-work/)
  - [Copyright](https://aclanthology.org/faq/copyright/)
  - * * *

  - [Credits](https://aclanthology.org/info/credits/)
  - [Volunteer](https://aclanthology.org/faq/volunteer/)
  - [Development](https://aclanthology.org/info/development/)
  - [Feedback](https://aclanthology.org/faq/feedback/)
- [Using](https://aclanthology.org/2025.findings-acl.428/#)
  - [Citing papers](https://aclanthology.org/faq/bib/)
  - [Links in the Anthology](https://aclanthology.org/faq/linking/)
  - [Data access](https://aclanthology.org/faq/api/)
  - * * *

  - [All FAQs](https://aclanthology.org/faq/)
  - * * *

  - ###### Details

  - [Anthology identifiers](https://aclanthology.org/info/ids/)
  - [Names](https://aclanthology.org/info/names/)
  - [ORCID iDs](https://aclanthology.org/info/orcid/)
  - [DOIs](https://aclanthology.org/faq/doi/)
  - [Verified authors](https://aclanthology.org/info/verification/)
- [Contributions](https://aclanthology.org/2025.findings-acl.428/#)
  - [Submissions](https://aclanthology.org/info/contrib/)
  - [Corrections](https://aclanthology.org/info/corrections/)
  - [Author pages](https://aclanthology.org/info/author-pages/)
  - [Attachments](https://aclanthology.org/faq/attachments/)
- [GitHub](https://github.com/acl-org/acl-anthology/)

## [CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation](https://aclanthology.org/2025.findings-acl.428.pdf)

[Qingyao Li](https://aclanthology.org/people/qingyao-li/),
[Xinyi Dai](https://aclanthology.org/people/xinyi-dai/unverified/),
[Xiangyang Li](https://aclanthology.org/people/xiangyang-li/),
[Weinan Zhang](https://aclanthology.org/people/weinan-zhang-ucl/),
[Yasheng Wang](https://aclanthology.org/people/yasheng-wang/unverified/),
[Ruiming Tang](https://aclanthology.org/people/ruiming-tang/),
[Yong Yu](https://aclanthology.org/people/yong-yu/)

##### Correct Metadata for

Use this form to create a GitHub issue with structured data describing the correction. You will need a GitHub account.
Once you create that issue, the correction will be reviewed by a staff member.

⚠️ Mobile Users: Submitting this form to create a new issue will only work with github.com, not the GitHub Mobile app.

**Important**: The Anthology treat PDFs as authoritative. Please use this form only to correct data
that is out of line with the PDF. See [our corrections\\
guidelines](https://aclanthology.org/info/corrections/) if you need to change the PDF.

TitleAdjust the title. Retain tags such as
<fixed-case>.

AuthorsAdjust author names and order to match the
PDF.

Add Author

AbstractCorrect abstract if needed. Retain XML formatting tags such as <tex-math>. You may use <b>...</b> for **bold**, <i>...</i> for _italic_, <u>...</u> for underline, <sc>...</sc> for small-caps, <tt>...<tt> for `typewriter text`, <url>...</url> for URLs, <a href=...> for hyperlinks, and <par/> for paragraph breaks.

Verification against PDFEnsure that the new title/authors match the snapshot below. (If there
is no snapshot or it is too small, consult [the PDF](https://aclanthology.org/2025.findings-acl.428/#).)

[![](https://aclanthology.org/2025.findings-acl.428/)](https://aclanthology.org/2025.findings-acl.428/#)

Authors concatenated from the text boxes above:

ALL author names match the snapshot above—including
middle initials, hyphens, and accents.

Create GitHub issue for staff review

* * *

##### Abstract

Code generation is a critical reasoning task for large language models (LLMs). Recent advancements have focused on optimizing the thought process of code generation, achieving significant improvements. However, such thought process lacks effective process supervision, making it hard to optimize the thoughts. Although Process Reward Models (PRMs) have been widely established in mathematical reasoning, building a code PRM is still not trivial for the gap between thoughts to code. In this paper, we propose CodePRM, a novel approach that leverages the code execution feedback to build a code PRM. Specifically, we first collect a large dataset of thought traces, where each thought step is labeled with their derived code’ pass rates, accompanied by the corresponding code snippets, and execution feedback. During training, we train a PRM to take both the reasoning process and code execution feedback as input to score individual thought steps, enabling it to leverage code execution results to distinguish between high-quality and low-quality thought steps. Finally, to use the PRM during inference, we develop a Generate-Verify-Refine (GVR) pipeline where the CodePRM serves as a process verifier to dynamically identify and correct errors in the thought process during code search. Experimental results demonstrate that CodePRM with the inference algorithm outperforms strong baselines, significantly enhancing code generation performance. Further analysis reveals the key factors for building a code PRM.

Anthology ID:2025.findings-acl.428Volume:[Findings of the Association for Computational Linguistics: ACL 2025](https://aclanthology.org/volumes/2025.findings-acl/)Month:JulyYear:2025Address:Vienna, AustriaEditors:[Wanxiang Che](https://aclanthology.org/people/wanxiang-che/),
[Joyce Nabende](https://aclanthology.org/people/joyce-nabende/unverified/),
[Ekaterina Shutova](https://aclanthology.org/people/ekaterina-shutova/unverified/),
[Mohammad Taher Pilehvar](https://aclanthology.org/people/mohammad-taher-pilehvar/unverified/)Venue:[Findings](https://aclanthology.org/venues/findings/ "Findings of the Association for Computational Linguistics")SIG:Publisher:Association for Computational LinguisticsNote:Pages:8169–8182Language:URL:[https://aclanthology.org/2025.findings-acl.428/](https://aclanthology.org/2025.findings-acl.428/)DOI:[10.18653/v1/2025.findings-acl.428](https://doi.org/10.18653/v1/2025.findings-acl.428 "To the current version of the paper by DOI")Bibkey:li-etal-2025-codeprmCite (ACL):Qingyao Li, Xinyi Dai, Xiangyang Li, Weinan Zhang, Yasheng Wang, Ruiming Tang, and Yong Yu. 2025. [CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation](https://aclanthology.org/2025.findings-acl.428/). In _Findings of the Association for Computational Linguistics: ACL 2025_, pages 8169–8182, Vienna, Austria. Association for Computational Linguistics.Cite (Informal):[CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation](https://aclanthology.org/2025.findings-acl.428/) (Li et al., Findings 2025)Copy Citation:BibTeXMarkdownMODS XMLEndnoteMore
options…PDF:[https://aclanthology.org/2025.findings-acl.428.pdf](https://aclanthology.org/2025.findings-acl.428.pdf)

[PDF](https://aclanthology.org/2025.findings-acl.428.pdf "Open PDF of 'CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation'") [Cite](https://aclanthology.org/2025.findings-acl.428/# "Open dialog for exporting citations") [Search](https://www.semanticscholar.org/search?+q=CodePRM%3A+Execution+Feedback-enhanced+Process+Reward+Model+for+Code+Generation "Search for 'CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation' on Semantic Scholar") [Fix data](https://aclanthology.org/2025.findings-acl.428/# "Correct problems with title, author list, and abstract")

* * *

##### Export citation

- [BibTeX](https://aclanthology.org/2025.findings-acl.428/#citeBibtex)
- [MODS XML](https://aclanthology.org/2025.findings-acl.428/#citeMods)
- [Endnote](https://aclanthology.org/2025.findings-acl.428/#citeEndnote)
- [Preformatted](https://aclanthology.org/2025.findings-acl.428/#citeMarkdown)

```
@inproceedings{li-etal-2025-codeprm,
    title = "{C}ode{PRM}: Execution Feedback-enhanced Process Reward Model for Code Generation",
    author = "Li, Qingyao  and
      Dai, Xinyi  and
      Li, Xiangyang  and
      Zhang, Weinan  and
      Wang, Yasheng  and
      Tang, Ruiming  and
      Yu, Yong",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.428/",
    doi = "10.18653/v1/2025.findings-acl.428",
    pages = "8169--8182",
    ISBN = "979-8-89176-256-5",
    abstract = "Code generation is a critical reasoning task for large language models (LLMs). Recent advancements have focused on optimizing the thought process of code generation, achieving significant improvements. However, such thought process lacks effective process supervision, making it hard to optimize the thoughts. Although Process Reward Models (PRMs) have been widely established in mathematical reasoning, building a code PRM is still not trivial for the gap between thoughts to code. In this paper, we propose CodePRM, a novel approach that leverages the code execution feedback to build a code PRM. Specifically, we first collect a large dataset of thought traces, where each thought step is labeled with their derived code' pass rates, accompanied by the corresponding code snippets, and execution feedback. During training, we train a PRM to take both the reasoning process and code execution feedback as input to score individual thought steps, enabling it to leverage code execution results to distinguish between high-quality and low-quality thought steps. Finally, to use the PRM during inference, we develop a Generate-Verify-Refine (GVR) pipeline where the CodePRM serves as a process verifier to dynamically identify and correct errors in the thought process during code search. Experimental results demonstrate that CodePRM with the inference algorithm outperforms strong baselines, significantly enhancing code generation performance. Further analysis reveals the key factors for building a code PRM."
}
```

Download as
FileCopy to Clipboard

```
<?xml version="1.0" encoding="UTF-8"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
<mods ID="li-etal-2025-codeprm">
    <titleInfo>
        <title>CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation</title>
    </titleInfo>
    <name type="personal">
        <namePart type="given">Qingyao</namePart>
        <namePart type="family">Li</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Xinyi</namePart>
        <namePart type="family">Dai</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Xiangyang</namePart>
        <namePart type="family">Li</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Weinan</namePart>
        <namePart type="family">Zhang</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Yasheng</namePart>
        <namePart type="family">Wang</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Ruiming</namePart>
        <namePart type="family">Tang</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Yong</namePart>
        <namePart type="family">Yu</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <originInfo>
        <dateIssued>2025-07</dateIssued>
    </originInfo>
    <typeOfResource>text</typeOfResource>
    <relatedItem type="host">
        <titleInfo>
            <title>Findings of the Association for Computational Linguistics: ACL 2025</title>
        </titleInfo>
        <name type="personal">
            <namePart type="given">Wanxiang</namePart>
            <namePart type="family">Che</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Joyce</namePart>
            <namePart type="family">Nabende</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Ekaterina</namePart>
            <namePart type="family">Shutova</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Mohammad</namePart>
            <namePart type="given">Taher</namePart>
            <namePart type="family">Pilehvar</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <originInfo>
            <publisher>Association for Computational Linguistics</publisher>
            <place>
                <placeTerm type="text">Vienna, Austria</placeTerm>
            </place>
        </originInfo>
        <genre authority="marcgt">conference publication</genre>
        <identifier type="isbn">979-8-89176-256-5</identifier>
    </relatedItem>
    <abstract>Code generation is a critical reasoning task for large language models (LLMs). Recent advancements have focused on optimizing the thought process of code generation, achieving significant improvements. However, such thought process lacks effective process supervision, making it hard to optimize the thoughts. Although Process Reward Models (PRMs) have been widely established in mathematical reasoning, building a code PRM is still not trivial for the gap between thoughts to code. In this paper, we propose CodePRM, a novel approach that leverages the code execution feedback to build a code PRM. Specifically, we first collect a large dataset of thought traces, where each thought step is labeled with their derived code’ pass rates, accompanied by the corresponding code snippets, and execution feedback. During training, we train a PRM to take both the reasoning process and code execution feedback as input to score individual thought steps, enabling it to leverage code execution results to distinguish between high-quality and low-quality thought steps. Finally, to use the PRM during inference, we develop a Generate-Verify-Refine (GVR) pipeline where the CodePRM serves as a process verifier to dynamically identify and correct errors in the thought process during code search. Experimental results demonstrate that CodePRM with the inference algorithm outperforms strong baselines, significantly enhancing code generation performance. Further analysis reveals the key factors for building a code PRM.</abstract>
    <identifier type="citekey">li-etal-2025-codeprm</identifier>
    <identifier type="doi">10.18653/v1/2025.findings-acl.428</identifier>
    <location>
        <url>https://aclanthology.org/2025.findings-acl.428/</url>
    </location>
    <part>
        <date>2025-07</date>
        <extent unit="page">
            <start>8169</start>
            <end>8182</end>
        </extent>
    </part>
</mods>
</modsCollection>
```

Download as
FileCopy to Clipboard

```
%0 Conference Proceedings
%T CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation
%A Li, Qingyao
%A Dai, Xinyi
%A Li, Xiangyang
%A Zhang, Weinan
%A Wang, Yasheng
%A Tang, Ruiming
%A Yu, Yong
%Y Che, Wanxiang
%Y Nabende, Joyce
%Y Shutova, Ekaterina
%Y Pilehvar, Mohammad Taher
%S Findings of the Association for Computational Linguistics: ACL 2025
%D 2025
%8 July
%I Association for Computational Linguistics
%C Vienna, Austria
%@ 979-8-89176-256-5
%F li-etal-2025-codeprm
%X Code generation is a critical reasoning task for large language models (LLMs). Recent advancements have focused on optimizing the thought process of code generation, achieving significant improvements. However, such thought process lacks effective process supervision, making it hard to optimize the thoughts. Although Process Reward Models (PRMs) have been widely established in mathematical reasoning, building a code PRM is still not trivial for the gap between thoughts to code. In this paper, we propose CodePRM, a novel approach that leverages the code execution feedback to build a code PRM. Specifically, we first collect a large dataset of thought traces, where each thought step is labeled with their derived code’ pass rates, accompanied by the corresponding code snippets, and execution feedback. During training, we train a PRM to take both the reasoning process and code execution feedback as input to score individual thought steps, enabling it to leverage code execution results to distinguish between high-quality and low-quality thought steps. Finally, to use the PRM during inference, we develop a Generate-Verify-Refine (GVR) pipeline where the CodePRM serves as a process verifier to dynamically identify and correct errors in the thought process during code search. Experimental results demonstrate that CodePRM with the inference algorithm outperforms strong baselines, significantly enhancing code generation performance. Further analysis reveals the key factors for building a code PRM.
%R 10.18653/v1/2025.findings-acl.428
%U https://aclanthology.org/2025.findings-acl.428/
%U https://doi.org/10.18653/v1/2025.findings-acl.428
%P 8169-8182
```

Download as
FileCopy to Clipboard

##### Markdown (Informal)

\[CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation\](https://aclanthology.org/2025.findings-acl.428/) (Li et al., Findings 2025)

- [CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation](https://aclanthology.org/2025.findings-acl.428/) (Li et al., Findings 2025)

##### ACL

- Qingyao Li, Xinyi Dai, Xiangyang Li, Weinan Zhang, Yasheng Wang, Ruiming Tang, and Yong Yu. 2025. [CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation](https://aclanthology.org/2025.findings-acl.428/). In _Findings of the Association for Computational Linguistics: ACL 2025_, pages 8169–8182, Vienna, Austria. Association for Computational Linguistics.

Copy Markdown to
ClipboardCopy ACL to
Clipboard

[![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
ACL materials are Copyright © 1963–2026 ACL; other materials are copyrighted by their respective copyright holders. Materials prior to 2016 here are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 International License](https://creativecommons.org/licenses/by-nc-sa/3.0/). Permission is granted to make copies for the purposes of teaching and research. Materials published in or after 2016 are licensed on a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

The ACL Anthology is managed and built by the [ACL Anthology team](https://aclanthology.org/info/credits/) of volunteers.

_Site last built on 25 July 2026 at 23:05 UTC with [commit 1a19147](https://github.com/acl-org/acl-anthology/tree/1a19147415245d2bae2620d00f247778d20147b7)._