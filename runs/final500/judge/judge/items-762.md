Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=80c5480cb63ce4f0 ===
Gold sentences and facts:
Sentence 0: Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters . The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
  Fact 0.0: Taxpayers in the United States may face various penalties for failures related to federal tax matters
  Fact 0.1: Taxpayers in the United States may face various penalties for failures related to state tax matters
  Fact 0.2: Taxpayers in the United States may face various penalties for failures related to local tax matters
  Fact 0.3: The Internal Revenue Service (IRS) is primarily responsible for charging these penalties at the federal level
  Fact 0.4: The IRS can assert only those penalties specified under federal tax law
  Fact 0.5: State and local tax penalty rules vary widely
  Fact 0.6: State and local tax penalty rules are administered by state and local authorities
  Fact 0.7: State and local tax penalty rules are not discussed in the source text

Retrieved items:
<item 1>
IRS penalties
Tax adviser penalties ( edit ) Penalties also apply to people who promote tax shelters or who fail to maintain and disclose lists of reportable transactions
their customers or clients for those transactions . These monetary penalties can be severe . Judicial appeal of penalties ( edit ) Most penalties are subject to
judicial review . However , the courts rarely modify assessment of the penalties and interest for underestimate or late payment . Further reading ( edit )
Meldman , Robert E. , Federal Taxation Practice and Procedure , ISBN 978 - 0 - 8080 - 1647 - 2 , Chapters 9 and 18 Shafiroff , Ira L. , Internal Revenue Service
Practice and Procedure Deskbook , Kindle edition ( no ISBN ) , or hardcover ISBN 978 - 0 - 87224 - 109 - 1 Pratt , Kulsrud , et al , Federal Taxation , 2006
Edition ISBN 978 - 0 - 7593 - 5175 - 2 , pages 2 -- 6 through 2 - 15 .
</item 1>

<item 2>
IRS penalties
Some of these taxes require purchase of tax stamps or other evidence of advance payment of tax . Some require collection of the tax by retailers . An assortment
of penalties apply to manufacturers and retailers not complying with the particular rules . In addition , certain penalties are imposed in the form of an excise
tax . Pension and benefit plans must pay a tax for a variety of failures . Charities and private foundations must pay an excise tax on prohibited transactions
and other failures . Tax fraud penalties ( edit ) Intentional filing of materially false tax returns is a criminal offence . A person convicted of committing
tax fraud , or aiding and abetting another in committing tax fraud , may be subject to forfeiture of property and / or jail time . Conviction and sentencing is
through the court system . Responsibility for prosecution falls to the U.S. Department of Justice , not the Internal Revenue Service . Penalties may be assessed
against tax protesters who raise arguments that income tax laws are not valid , or who otherwise file frivolous returns or court petitions .
</item 2>

<item 3>
IRS penalties
Penalties may be monetary or may involve forfeiture of property . Criminal penalties may include jail time , but are imposed only by a federal judge after a
defendant is convicted . Most monetary penalties are based on the amount of tax not properly paid . Penalties may increase with the period of nonpayment . Some
penalties are fixed dollar amounts or fixed percentages of some measure required to be reported . Excise taxes used as penalties are imposed in the Code
sections relating to particular kinds of transactions . Some penalties may be waived or abated where the taxpayer shows reasonable cause for the failure .
Penalties apply for failures to file income tax returns or information returns , or for filing incorrect returns . Some penalties may be very minor . Penalties
apply for certain types of errors on tax returns , and may be substantial . Some penalties are imposed as excise taxes on particular transactions . Certain
other penalties apply for other types of failures . Certain acts may result in forfeiture of property of the taxpayer . There are over 150 kinds of civil
penalties in the U.S. Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines .
</item 3>

<item 4>
IRS penalties
Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines . Contents ( hide ) 1 Underestimate and late payment
penalties 2 Penalties for failure to timely file returns or timely pay tax 3 Accuracy related penalties 4 Penalties in connection with information returns 5 100
% penalty on unpaid withholding taxes 6 Penalties for failure to provide foreign information 7 Excise taxes as penalties 8 Tax fraud penalties 9 Tax adviser
penalties 10 Judicial appeal of penalties 11 Further reading 12 References Underestimate and late payment penalties ( edit ) Taxpayers are required to have
withholding of tax or make quarterly estimated tax payments before the end of the tax year . Because accurate estimation requires accurate prediction of the
future , taxpayers sometimes underestimate the amount due . The penalty for paying too little estimated tax or having too little tax withheld is computed with
interest on the amount that should have been , but was not , paid . Where a taxpayer has filed an income or excise tax return that shows a balance due but does
not pay that balance by the due date of the return ( without extensions ) , a different charge applies .
</item 4>

<item 5>
IRS penalties
Willis , Hoffman , et al , South - Western Federal Taxation , 2009 Edition ISBN 978 - 0 - 324 - 66052 - 4 , pages 26 -- 13 through 26 - 19 . References ( edit )
Jump up ^ Most of the penalty rules are in Chapters 68 through 75 of the Internal Revenue Code , sections 6651 through 7344 . 26 USC Subtitle F . Jump up ^ Alan
J. Tarr , J.D. , LL. M. ( Taxation ) & Pamela Jensen Drucker , J.D. , LL. M. ( Taxation ) , Civil Tax Penalties , U.S. Income Portfolios , Vol. 634 ( 2d ed.
2012 ) , Bloomberg BNA . Jump up ^ 26 USC 6654 . Jump up ^ 26 USC 6651 ( a ) ( 3 ) . Jump up ^ See IRS Tax Topic 653 . Jump up ^ 26 USC 6651 ( a ) ( 1 ) .
</item 5>

<item 6>
IRS penalties
This penalty of 20 % or 40 % of the increase in tax is due in the case of substantial understatement of tax , substantial valuation misstatements , transfer
pricing adjustments , or negligence or disregard of rules or regulations . For example , a valuation overstatement can result in a 30 % penalty on the amount of
tax owed . Special rules apply for each of these types of errors under which the penalty may be waived . Penalties in connection with information returns ( edit
) Certain types of returns , called `` information returns , '' do not require payment of tax . These include forms filed by employers to report wages ( Form W
- 2 ) and businesses to report certain payments ( Form 1099 series instructions ) . The penalty for failures related to these forms is a dollar amount per form
not timely filed , and the amount of penalty increases with the degree of lateness . The current maximum penalty for these forms is $50 . Many of the forms must
be filed electronically , and filing on paper is considered non-filing . Late filing of returns of partnership income ( Form 1065 ) can result in penalties of
$195 per month per partner , up to a maximum of 12 months .
</item 6>

<item 7>
IRS penalties
IRS penalties - wikipedia IRS penalties Jump to : navigation , search This article is part of a series on Taxation in the United States of America Federal
taxation ( show ) Alternative minimum tax Capital gains tax Corporate tax Estate tax Excise tax Gift tax Income tax Payroll tax Internal Revenue Service ( IRS )
Internal Revenue Code ( IRC ) IRS tax forms Revenue by state History Constitutional authority Taxpayer standing Court Protest Evasion State and local taxation (
show ) State income tax Property tax Sales tax Use tax Land value tax State tax levels Federal tax reform ( show ) Automated payment transaction tax 9 -- 9 -- 9
Competitive Tax Plan Efficient Taxation of Income FairTax Flat tax Hall -- Rabushka flat tax Kemp Commission Taxpayer Choice Act USA Tax Value added tax Border
- adjustment tax United States portal Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters
. The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties
specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
</item 7>

<item 8>
IRS penalties
Similar penalties may apply to an income tax return ( Form 1120S ) for an S corporation return . 100 % penalty on unpaid withholding taxes ( edit ) Employers
are required to withhold income and social security taxes from wages paid to employees , and to pay these amounts promptly to the government . A penalty of 100
% of the amount not paid over ( plus liability for paying the withheld amounts ) may be collected without judicial proceedings from each and every person who
had custody and control of the funds and did not make the payment to the government . This applies to company employees and officers as individuals , as well as
to companies themselves . There have been reported cases of the IRS seizing houses of those failing to pay over employee taxes . Penalties for failure to
provide foreign information ( edit ) Taxpayers who are shareholders of controlled foreign corporations must file Form 5471 with respect to each such controlled
foreign corporation . Penalties for failure to timely file are $10,000 to $50,000 per form , plus possible loss of foreign tax credits . U.S.
</item 8>

<item 9>
IRS penalties
The late filing penalty may be waived or abated on showing of reasonable cause for failure . The failure to file penalty is imposed and starts to accrue
interest from the due date of the return . The failure to pay penalty is imposed when a taxpayer pays the taxes after payment was due , computed from the date
prescribed for paying the tax . The Internal Revenue Service advises that if the taxpayer wants to compute the penalty for failure to timely file and the
penalty for failure to timely pay the tax shown on the return , or the interest , and to pay those items at the time the return is filed , the taxpayer can ``
identify and enter the amount in the bottom margin '' on the second page of Form 1040 . The IRS advises that the taxpayer `` not include interest or penalties (
other than the estimated tax penalty ) '' in the `` Amount Owed '' line of the form . Accuracy related penalties ( edit ) If amounts reported on an income tax
return are later adjusted by the IRS and a tax increase results , an additional penalty may apply .
</item 9>

<item 10>
IRS penalties
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 10>
=== END ITEM id=80c5480cb63ce4f0 ===

=== ITEM id=394f85cb77a0774e ===
Gold sentences and facts:
Sentence 0: A regulatory site ( also known as an `` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes conformational changes which changes the enzyme 's activity .
  Fact 0.0: A regulatory site is also known as an allosteric site.
  Fact 0.1: A regulatory site is a site on an allosteric protein.
  Fact 0.2: A modulator molecule binds to a regulatory site on an allosteric protein.
  Fact 0.3: A modulator molecule is also known as an allosteric regulator.
  Fact 0.4: The ligand-binding site on a receptor or enzyme is distinct from the active site.
  Fact 0.5: Allosteric modulators alter enzyme activity by binding to the regulatory site.
  Fact 0.6: Some allosteric regulators activate enzymes.
  Fact 0.7: Other allosteric regulators inhibit enzymes.
  Fact 0.8: The working hypothesis is that when allosteric regulators bind to the regulatory site, the protein undergoes conformational changes.
  Fact 0.9: The working hypothesis is that these conformational changes change the enzyme's activity.

Retrieved items:
<item 1>
Regulatory site
Regulatory site - Wikipedia Regulatory site Jump to : navigation , search It has been suggested that this article be merged with Allosteric regulation . (
Discuss ) Proposed since October 2017 . This article does not cite any sources . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to remove this template message ) A regulatory site ( also known as an
`` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a
receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators
activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes
conformational changes which changes the enzyme 's activity . This biochemistry article is a stub . You can help Wikipedia by expanding it .
</item 1>

<item 2>
ATP synthase
Jump up ^ Ahmad Z , Okafor F , Laughlin TF . `` Role of Charged Residues in the Catalytic Sites of Escherichia coli ATP Synthase '' . Journal of Amino Acids.
2011 : 785741 . doi : 10.4061 / 2011 / 785741 . PMID 22312470 . Jump up ^ Devenish RJ , Prescott M , Roucou X , Nagley P ( May 2000 ) . `` Insights into ATP
synthase assembly and function through the molecular genetic manipulation of subunits of the yeast mitochondrial enzyme complex '' . Biochimica et Biophysica
Acta. 1458 ( 2 -- 3 ) : 428 -- 42 . doi : 10.1016 / S0005 - 2728 ( 00 ) 00092 - X . PMID 10838056 . Jump up ^ Kabaleeswaran V , Puri N , Walker JE , Leslie AG ,
Mueller DM ( November 2006 ) . `` Novel features of the rotary catalytic mechanism revealed in the structure of yeast F ATPase '' .
</item 2>

<item 3>
ATP synthase
5.5 - 6 : Polymerization motors Dynamin Tubulin Enzymes Activity Active site Binding site Catalytic triad Oxyanion hole Enzyme promiscuity Catalytically perfect
enzyme Coenzyme Cofactor Enzyme catalysis Regulation Allosteric regulation Cooperativity Enzyme inhibitor Classification EC number Enzyme superfamily Enzyme
family List of enzymes Kinetics Enzyme kinetics Eadie -- Hofstee diagram Hanes -- Woolf plot Lineweaver -- Burk plot Michaelis -- Menten kinetics Types EC1
Oxidoreductases ( list ) EC2 Transferases ( list ) EC3 Hydrolases ( list ) EC4 Lyases ( list ) EC5 Isomerases ( list ) EC6 Ligases ( list ) Molecular and
Cellular Biology portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=ATP_synthase&oldid=821081799 '' Categories : Enzymes Cellular respiration
Photosynthesis EC 3.6.
</item 3>

<item 4>
ATP synthase
The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right
conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane . The binding change
mechanism involves the active site of a β subunit 's cycling between three states . In the `` loose '' state , ADP and phosphate enter the active site ; in the
adjacent diagram , this is shown in pink . The enzyme then undergoes a change in shape and forces these molecules together , with the active site in the
resulting `` tight '' state ( shown in red ) binding the newly produced ATP molecule with very high affinity . Finally , the active site cycles back to the open
state ( orange ) , releasing ATP and binding more ADP and phosphate , ready for the next cycle of ATP production . Physiological role ( edit ) Like other
enzymes , the activity of F F ATP synthase is reversible .
</item 4>

<item 5>
ATP synthase
`` Atomic model for the dimeric FO region of mitochondrial ATP synthase '' . Science . 358 ( 6365 ) : 936 -- 40 . doi : 10.1126 / science. aao4815 . PMID
29074581 . Jump up ^ Gresser MJ , Myers JA , Boyer PD ( October 1982 ) . `` Catalytic site cooperativity of beef heart mitochondrial F adenosine triphosphatase
. Correlations of initial velocity , bound intermediate , and oxygen exchange measurements with an alternating three - site model '' . The Journal of Biological
Chemistry . 257 ( 20 ) : 12030 -- 8 . PMID 6214554 . Jump up ^ Nakamoto RK , Baylis Scanlon JA , Al - Shawi MK ( August 2008 ) . `` The rotary mechanism of the
ATP synthase '' . Archives of Biochemistry and Biophysics. 476 ( 1 ) : 43 -- 50 . doi : 10.1016 / j. abb. 2008.05. 004 . PMC 2581510 .
</item 5>

<item 6>
ATP synthase
Simplified model of F F - ATPase alias ATP synthase of E. coli . Subunits of the enzyme are labeled accordingly . Rotation engine of ATP synthase . Located
within the thylakoid membrane and the inner mitochondrial membrane , ATP synthase consists of two regions F and F. F causes rotation of F and is made of c -
ring and subunits a , b , d , F6. F is made of α , β , γ , δ ( \ displaystyle \ alpha , \ beta , \ gamma , \ delta ) subunits . F has a water soluble par that
can hydrolyze ATP . F on the other hand has mainly hydrophobic regions . F F creates a pathway for protons movement across the membrane . F region ( edit ) The
F portion of ATP synthase is hydrophilic and responsible for hydrolyzing ATP . Subunits α ( \ displaystyle \ alpha ) and β ( \ displaystyle \ beta ) make a
hexamer with 6 binding sites . Three of them are catalytically inactive and they bind ADP . Other three subunits catalyze the ATP synthesis .
</item 6>

<item 7>
Regulatory site
This biochemistry article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Regulatory_site&oldid=807494796 '' Categories : Protein structure Biochemistry stubs Hidden categories : Articles to
be merged from October 2017 All articles to be merged Articles lacking sources from December 2009 All articles lacking sources All stub articles Talk About
Wikipedia Add links This page was last edited on 28 October 2017 , at 09 : 38 . About Wikipedia
</item 7>

<item 8>
ATP synthase
Proton and Sodium translocating F - type , V - type and A-type ATPases in OPM database The Nobel Prize in Chemistry 1997 to Paul D. Boyer and John E. Walker for
the enzymatic mechanism of synthesis of ATP ; and to Jens C. Skou , for discovery of an ion - transporting enzyme , Na + , K + - ATPase . Harvard Multimedia
Production Site -- Videos -- ATP synthesis animation David Goodsell : `` ATP Synthase - Molecule of the Month '' Membrane transport protein : ion pumps ,
ATPases / ATP synthase ( TC 3A2 - 3A3 ) F - , V - , and A-type ATPase ( 3. A.
</item 8>

<item 9>
ATP synthase
ATP synthase - wikipedia ATP synthase Jump to : navigation , search ATP Synthase Molecular model of ATP synthase determined by X-ray crystallography . Stator is
not shown here . Identifiers EC number 3.6. 3.14 CAS number 9000 - 83 - 3 Databases IntEnz IntEnz view BRENDA BRENDA entry ExPASy NiceZyme view KEGG KEGG entry
MetaCyc metabolic pathway PRIAM profile PDB structures RCSB PDB PDBe PDBsum Gene Ontology AmiGO / QuickGO ( show ) Search PMC articles PubMed articles NCBI
proteins ATP synthase is an enzyme that creates the energy storage molecule adenosine triphosphate ( ATP ) . ATP is the most commonly used `` energy currency ''
of cells for most organisms . It is formed from adenosine diphosphate ( ADP ) and inorganic phosphate ( P ) .
</item 9>

<item 10>
ATP synthase
`` Partial resolution of the enzymes catalyzing oxidative phosphorylation. 8 . Properties of a factor conferring oligomycin sensitivity on mitochondrial
adenosine triphosphatase '' . The Journal of Biological Chemistry . 241 ( 10 ) : 2461 -- 6 . PMID 4223640 . Jump up ^ Mccarty RE ( November 1992 ) . `` A plant
biochemist 's view of H + - ATPases and ATP synthases `` . The Journal of Experimental Biology . 172 ( Pt 1 ) : 431 -- 441 . PMID 9874753 . Jump up ^ PDB : 5ARA
​ ; Zhou A , Rohou A , Schep DG , Bason JV , Montgomery MG , Walker JE , Grigorieff N , Rubinstein JL ( October 2015 ) . `` Structure and conformational states
of the bovine mitochondrial ATP synthase by cryo - EM '' . eLife. 4 : e10180 . doi : 10.7554 / eLife. 10180 . PMC 4718723 . PMID 26439008 .
</item 10>
=== END ITEM id=394f85cb77a0774e ===

=== ITEM id=3caca3ff6113624d ===
Gold sentences and facts:
Sentence 0: Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995
  Fact 0.0: Mumbai has the Marathi name मुंबई.
  Fact 0.1: Mumbai was previously named Bombay.
  Fact 0.2: Mumbai was renamed in 1995.

Retrieved items:
<item 1>
Renaming of cities in India
Jump up ^ Calcutta : A Cultural and Literary History - Page 3 Krishna Dutta - 2003 `` nationalist stance , like Bombay , which changed its name to Mumbai , or
Madras , which has become the unrecognisable Chennai , Calcutta has preferred a comparatively minor name change , which frankly is a bit of a multicultural
mishmash . '' Jump up ^ `` David Rumsey : Geographical Searching with MapRank Search ( beta ) '' . Rumsey.mapranksearch.com . Retrieved 2012 - 08 - 15 . Jump up
^ Beam , Christopher ( 2006 - 07 - 12 ) . `` How Bombay became Mumbai . - Slate Magazine '' . Slate.com . Retrieved 2012 - 08 - 15 .
</item 1>

<item 2>
Renaming of cities in India
from Trivandrum , respelled in 1991 Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995 Kochi ( Malayalam : കൊച്ചി ) , from Cochin , respelled in 1996
Chennai ( Tamil : சென்னை ) , from Madras , renamed in 1996 Kolkata ( Bengali : কলকাতা ) , from Calcutta , respelled in 2001 Kadapa ( Telugu : కడప ) , from
Cuddapah , respelled in 2005 Puducherry ( Tamil : புதுச்சேரி ) , from Pondicherry , renamed in 2006 Bengaluru ( Kannada : ಬೆಂಗಳೂರು ) , from Bangalore ,
</item 2>

<item 3>
Renaming of cities in India
It is an indigenizing toponymic '' Jump up ^ Reserve Bank of India 's instructions for banks & banking operations Reserve Bank of India 2001 Page 713 `` The new
name `` Mumbai '' should be reflected in both English and Hindi and the change in name is to be brought about in all official communications , name plates ,
sign boards , office seals , rubber stamps , etc . '' Jump up ^ Perveez Mody The Intimate State : Love - Marriage and the Law in Delhi Page 59 - 2008 ``
Throughout this book , I refer to India 's commercial capital as Bombay rather than Mumbai ... I am well aware of the name - change effected by an Act of the
Indian Parliament in 1997 that made the city officially ' Mumbai ' ... It is the same convention I adopt when referring to Calcutta rather than Kolkata . ''
Jump up ^ Pingali Sailaja Indian English Page 16 2009 `` Bombay is now called Mumbai , Madras is now Chennai and Calcutta is Kolkata , in an attempt to
de-anglicise them . In this work , the earlier names are retained since these names were used during the period that we mostly cover . ''
</item 3>

<item 4>
Renaming of cities in India
The name changes are not without controversy among Indians . In several instances , the name change represents a struggle between a cosmopolitan elite and a
local , regional - language populace over defining the city in ways that go far beyond a simple change of name . '' Jump up ^ Steven I. Wilkinson Votes and
Violence : Electoral Competition and Ethnic Riots in India 2006 Page 23 `` The BJP proposed in 1990 and 2001 that Ahmedabad be renamed `` Karnavati . '' Hindu ,
June 1 1 , 2001 . Similar proposals have been made to rename Allahabad `` Prayag . '' Jump up ^ Cosmopolitanism - Page 73 Carol A. Breckenridge , Sheldon
Pollock , Homi K. Bhabha - 2002 `` In one sense , the decision to officialise the name Mumbai is part of a widespread Indian pattern of replacing names
associated with colonial rule with names associated with local , national , and regional heroes .
</item 4>

<item 5>
Renaming of cities in India
For example , the change of Madras ( Tamil : மதராஸ் Madras ) to Chennai ( Tamil : சென்னை Chennai ) was reflected in many of India 's languages , and
incidentally in English , while the Tamil endonym had always been Chennai and remained unaffected by the change . Renaming in English ( edit ) Change in
official English spelling ( edit ) The renaming of cities is often specifically from English to Indian English in connection with that dialect 's internal
reforms . In other words , the city itself is not actually renamed in the local language , and the local name ( or endonym ) in the indigenous languages of
India does not change , but the official spelling in Indian English is amended . An example is the change from English `` Calcutta '' to English `` Kolkata ''
-- the local Bengali name ( কলকাতা Kôlkata ) did not change . Such changes in English spelling may be in order to better reflect a more accurate phonetic
transliteration of the local name , or may be for other reasons .
</item 5>

<item 6>
Renaming of cities in India
Renaming of cities in India - Wikipedia Renaming of cities in India Jump to : navigation , search The renaming of cities in India started in 1947 following the
end of the British imperial period . Several changes were politically controversial , and not all proposed changes were implemented . Each had to be approved by
the central government in New Delhi . The renaming of states and territories in India has also taken place , but until recently with actual substantial name
changes in both local language and in English such as the old British state name of Travancore - Cochin to Kerala ( 1956 ) . The most notable recent exceptions
are Indian English spelling - changes of Orissa to Odisha ( March 2011 ) and the Union Territory of Pondicherry ( which includes the City of Pondicherry ) to
Puducherry .
</item 6>

<item 7>
Renaming of cities in India
In the early years after Indian independence , many name changes were effected in northern India for English spellings of Hindi place names that had simply been
romanised inconsistently by the British administration -- such as the British spelling `` Jubbulpore , '' renamed `` Jabalpur '' ( जबलपुर ) among the first
changes in 1947 . These changes did not generate significant controversy . More recent and high - profile changes -- including renaming such major cities as
Calcutta to Kolkata -- have generated greater controversy . Since independence , such changes have typically been enacted officially by legislation at local or
national Indian government level , and may or may not then be adopted by the Indian media , particularly the influential Indian press . In the case of smaller
towns and districts which were less notable outside and inside India , and where a well known English name ( or exonym ) could not be said to exist , older
spellings used under British India may not have had any specific legislation other than changes in practice on the romanisation of indigenous Indian language
names .
</item 7>

<item 8>
Renaming of cities in India
Madhya Bharat → Madhya Pradesh ( 1 November 1959 ) Madras State → Tamil Nadu ( 14 January 1969 ) Mysore State → Karnataka ( 1 November 1973 ) Uttaranchal →
Uttarakhand ( 1 January 2007 ) Orissa → Odisha ( 4 November 2011 ) Cities ( edit ) Notable city names that were officially changed by legislation after
independence include : Jabalpur ( Hindi : जबलपुर ) , from Jubbulpore , respelled in 1947 Jajmau ( Hindi : जाजमऊ ) , from Jajesmow , respelled in 1948 Kanpur (
Hindi : कानपुर ) , from Cawnpore , respelled in 1948 Vadodara ( Gujarati : વડોદરા ) , from Baroda , respelled in 1974 Thiruvananthapuram ( Malayalam :
തിരുവനന്തപുരം ) , from Trivandrum ,
</item 8>

<item 9>
Renaming of cities in India
Realignment of the official Indian English name to an alternative local name ( edit ) Aside from changes to the official English spellings of local names there
have also been renaming proposals to realign the official name , hence the English name with an alternative local name . Ethnically sensitive examples include
the proposals by the Bharatiya Janata Party ( 1990 , 2001 ) to rename Ahmedabad to Karnavati and Allahabad to Prayag . These two proposals are changes from the
Muslim ( or Mughal ) name to a Hindu native name . These can be represented as a change from Urdu language to Hindi language , but since the two languages are
variants of Hindustani the proposal is effectively a cultural and ethno - religious proposal rather than a linguistic one . Adoption of name changes in India
and overseas ( edit ) Official name changes naturally take place quickly if not immediately in official government sources . Adoption may be slower among the
media in India and abroad , and among Indian authors . Important examples ( edit ) Main article : List of renamed places in India States ( edit ) Travancore -
Cochin → Kerala ( 1 November 1956 ) .
</item 9>

<item 10>
Renaming of cities in India
Other changes were only changes in some of the indigenous languages . For example , the renaming of the Madras Presidency to Madras State in 1947 and then Tamil
Nadu in 1969 required non-Tamil speakers to change from an approximation of the British name ( Tamil : மதராஸ் மாகாணம் Madras Presidency , then Madras State
Tamil : மதராஸ் மாநிலம் ) to a native Tamil name ( Tamil : தமிழ்நாடு Tamil Nadu , `` Tamil land '' ) . In general , changes to the local names of cities in the
indigenous languages are less common . However , a change in English may sometimes also be a reflection of changes in other Indian languages other than the
specific local one .
</item 10>
=== END ITEM id=3caca3ff6113624d ===

=== ITEM id=20da5a6e38e9f3d3 ===
Gold sentences and facts:
Sentence 0: The Third Five - year Plan , stressed agriculture and improvement in the production of wheat , but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army . In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat .
  Fact 0.0: The Third Five-year Plan stressed agriculture.
  Fact 0.1: The Third Five-year Plan stressed improvement in the production of wheat.
  Fact 0.2: The brief Sino-Indian War of 1962 exposed weaknesses in the economy.
  Fact 0.3: The brief Sino-Indian War of 1962 shifted the focus towards the defence industry and the Indian Army.
  Fact 0.4: India fought a war with Pakistan in 1965-1966.
  Fact 0.5: There was a severe drought in 1965.
  Fact 0.6: The war led to inflation.
  Fact 0.7: The priority was shifted to price stabilisation.
  Fact 0.8: The construction of dams continued.
  Fact 0.9: Many cement plants were built.
  Fact 0.10: Many fertilizer plants were built.
  Fact 0.11: Punjab began producing an abundance of wheat.

Retrieved items:
<item 1>
Five-Year Plans of India
Hydroelectric power projects and five steel plants at Bhilai , Durgapur , and Rourkela were established with the help of Russia , Britain ( the U.K ) and West
Germany respectively . Coal production was increased . More railway lines were added in the north east . The Tata Institute of Fundamental Research and Atomic
Energy Commission of India was established as research institutes . In 1957 a talent search and scholarship program was begun to find talented young students to
train for work in nuclear power . The total amount allocated under the Second Five - Year Plan in India was Rs. 48 billion . This amount was allocated among
various sectors : power and irrigation , social services , communications and transport , and miscellaneous . `` The target growth rate was 4.5 % and the actual
growth rate was 4.27 % . '' Third Plan ( 1961 -- 1966 ) ( edit ) The Third Five - year Plan , stressed agriculture and improvement in the production of wheat ,
but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army .
</item 1>

<item 2>
Five-Year Plans of India
The most important feature of this phase was active role of state in all economic sectors . Such a role was justified at that time because immediately after
independence , India was facing basic problems -- deficiency of capital and low capacity to save . The target growth rate was 2.1 % annual gross domestic
product ( GDP ) growth ; the achieved growth rate was 3.6 % the net domestic product went up by 15 % . The monsoon was good and there were relatively high crop
yields , boosting exchange reserves and the per capita income , which increased by 8 % . National income increased more than the per capita income due to rapid
population growth . Many irrigation projects were initiated during this period , including the Bhakra , Hirakud , Mettur Dam and Damodar Valley dams . The World
Health Organization ( WHO ) , with the Indian government , addressed children 's health and reduced infant mortality , indirectly contributing to population
growth . At the end of the plan period in 1956 , five Indian Institutes of Technology ( IITs ) were started as major technical institutions .
</item 2>

<item 3>
Five-Year Plans of India
Indian Economy : Problems of development and planning . pune : Wishwa Prakashan . p. 676 . Jump up ^ `` National Development Council approves 12th Five Year
Plan '' . Indian Express . 2012 - 12 - 27 . Retrieved 2013 - 07 - 10 . Jump up ^ ( 1 ) External links ( edit ) Official website of the Planning Commission of
India India topics History Overviews Timeline Years Astronomy Economics Linguistics Maritime Mathematics Metallurgy Military Postal Science and technology
Pre-colonial Stone Age Indus Valley Civilization Vedic period Mahajanapadas Mauryas Middle kingdoms Hoysala Chola Pala Kakatiya Delhi Sultanate Vijayanagara
Mughals Marathas European trade Colonial Princely East India Company Plassey 1857 rebellion British Raj Railways Economy Army Zamindari Bengali renaissance
Political reforms Princely states Partition of Bengal Independence movement 1943 famine World War II Partition Republic Integration Non-Aligned Movement Five -
Year Plans Sino - Indian War Indo - Pakistani wars Green Revolution White Revolution Naxal Insurgency Smiling Buddha Space programme The Emergency Indian Peace
Keeping Force ( IPKF ) Economic
</item 3>

<item 4>
Five-Year Plans of India
Contents ( hide ) 1 History 2 First Plan ( 1951 -- 1956 ) 3 Second Plan ( 1956 -- 1961 ) 4 Third Plan ( 1961 -- 1966 ) 5 Fourth Plan ( 1969 -- 1974 ) 6 Fifth
Plan ( 1974 -- 1978 ) 7 Rolling Plan ( 1978 -- 1980 ) 8 Sixth Plan ( 1980 -- 1985 ) 9 Seventh Plan ( 1985 -- 1990 ) 10 Annual Plans ( 1990 -- 1992 ) 11 Eighth
Plan ( 1992 -- 1997 ) 12 Ninth Plan ( 1997 -- 2002 ) 13 Tenth Plan ( 2002 -- 2007 ) 14 Eleventh Plan ( 2007 -- 2012 ) 15 Twelfth Plan ( 2012 -- 2017 ) 16 See
also 17 References 18 External links History ( edit ) Five - Year Plans ( FYPs ) are centralized and integrated national economic programs . Joseph Stalin
implemented the first Five Year Plan in the Soviet Union in the late 1920s .
</item 4>

<item 5>
Five-Year Plans of India
Most communist states and several capitalist countries subsequently have adopted them . China and India both continue to use FYPs , although China renamed its
Eleventh FYP , from 2006 to 2010 , a guideline ( guihua ) , rather than a plan ( jihua ) , to signify the central government 's more hands - off approach to
development . India launched its First FYP in 1951 , immediately after independence under socialist influence of first Prime Minister Jawaharlal Nehru . The
First Five - Year Plan was one of the most important because it had a great role in the launching of Indian development after Independence . Thus , it strongly
supported agriculture production and it also launched the industrialization of the country ( but less than the Second Plan , which focused on heavy industries )
. It built a particular system of mixed economy , with a great role for the public sector ( with an emerging welfare state ) , as well as a growing private
sector ( represented by some personalities as those who published the Bombay Plan ) . First Plan ( 1951 -- 1956 ) ( edit ) This article does not cite any
sources . Please help improve this article by adding citations to reliable sources .
</item 5>

<item 6>
Five-Year Plans of India
During 1966 -- 67 there was again the problem of drought . Equal priority was given to agriculture , its allied activities , and industrial sector . The
government of India declared `` Devaluation of Rupee '' to increase the exports of the country . The main reasons for plan holidays were the war , lack of
resources , and increase in inflation after that plan holiday was created . Fourth Plan ( 1969 -- 1974 ) ( edit ) At this time Indira Gandhi was the Prime
Minister . The Indira Gandhi government nationalised 14 major Indian banks and the Green Revolution in India advanced agriculture . In addition , the situation
in East Pakistan ( now Bangladesh ) was becoming dire as the Indo - Pakistan War of 1971 and Bangladesh Liberation War took funds earmarked for industrial
development . India also performed the Smiling Buddha underground nuclear test ( Pokhran - 1 ) in Rajasthan on May 18 , 1974 , partially in response to the
United States deployment of the Seventh Fleet in the Bay of Bengal . The fleet had been deployed to warn India against attacking West Pakistan and extending the
war .
</item 6>

<item 7>
Five-Year Plans of India
In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price
stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat . Many
primary schools were started in rural areas . In an effort to bring democracy to the grass - root level , Panchayat elections were started and the states were
given more development responsibilities . State electricity boards and state secondary education boards were formed . States were made responsible for secondary
and higher education . State road transportation corporations were formed and local road building became a state responsibility . The target growth rate was 5.6
% , but the actual growth rate was 2.4 % . Due to miserable failure of the Third Plan the government was forced to declare `` plan holidays '' ( from 1966 -- 67
, 1967 -- 68 , and 1968 -- 69 ) . Three annual plans were drawn during this intervening period . During 1966 -- 67 there was again the problem of drought .
</item 7>

<item 8>
Five-Year Plans of India
The fleet had been deployed to warn India against attacking West Pakistan and extending the war . The target growth rate was 5.6 % , but the actual growth rate
was 3.3 % . Fifth Plan ( 1974 -- 1978 ) ( edit ) The Fifth Five - Year Plan laid stress on employment , poverty alleviation ( Garibi Hatao ) , and justice . The
plan also focused on self - reliance in agricultural production and defence . In 1978 the newly elected Morarji Desai government rejected the plan . The
Electricity Supply Act was amended in 1975 , which enabled the central government to enter into power generation and transmission . The Indian national highway
system was introduced and many roads were widened to accommodate the increasing traffic . Tourism also expanded . The twenty - point programme was launched in
1975 . It was followed from 1974 to 1979 . The Minimum Needs Programme ( MNP ) was introduced in the first year of the Fifth Five Year Plan ( 1974 -- 78 ) . The
objective of the programme is to provide certain basic minimum needs and thereby improve the living standards of the people .
</item 8>

<item 9>
Five-Year Plans of India
The main advantage of the rolling plans is that they are flexible and are able to overcome the rigidity of fixed five year plans by mending targets , the object
of the exercise , projections and allocations as per the changing conditions in the country 's economy . The main disadvantage of this plan is that if the
targets are revised each year , it becomes very difficult to achieve them which are laid down in the five - year period and it turned out to be a complex plan .
Frequent revisions resulted in lack of stability in the economy which is essential for its balanced development and progress . Sixth Plan ( 1980 -- 1985 ) (
edit ) The Sixth Five - Year Plan marked the beginning of economic liberalisation . Price controls were eliminated and ration shops were closed . This led to an
increase in food prices and an increase in the cost of living . This was the end of Nehruvian socialism . The National Bank for Agriculture and Rural
Development was established for development of rural areas on 12 July 1982 by recommendation of the Shivaraman Committee . Family planning was also expanded in
order to prevent overpopulation .
</item 9>

<item 10>
Five-Year Plans of India
Five - Year plans of India - Wikipedia Five - Year plans of India Jump to : navigation , search From 1947 to 2017 , the Indian economy was premised on the
concept of planning . This was carried through the Five - Year Plans , developed , executed , and monitored by the Planning Commission ( 1951 - 2014 ) and the
NITI Aayog ( 2015 - 2017 ) . With the prime minister as the ex-officio chairman , the commission has a nominated deputy chairman , who holds the rank of a
cabinet Minister . Montek Singh Ahluwalia is the last deputy chairman of the commission ( resigned on 26 May 2014 ) . The Twelfth Plan completed its term in
March 2017 . Prior to the Fourth Plan , the allocation of state resources was based on schematic patterns rather than a transparent and objective mechanism ,
which led to the adoption of the Gadgil formula in 1969 . Revised versions of the formula have been used since then to determine the allocation of central
assistance for state plans .
</item 10>
=== END ITEM id=20da5a6e38e9f3d3 ===

=== ITEM id=a2a622b1158dd55c ===
Gold sentences and facts:
Sentence 0: The Director of the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the Mint has been Acting Principal Deputy Director David Motl .
  Fact 0.0: The Director of the United States Mint is the head of the United States Mint.
  Fact 0.1: The position of Director of the United States Mint is currently vacant.
  Fact 0.2: Since January 20, 2017, the senior career official at the Mint has been David Motl.
  Fact 0.3: David Motl is the Acting Principal Deputy Director.
Sentence 1: Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April 1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February 1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922 September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H. Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W. Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C. Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump
  Fact 1.0: David Rittenhouse held the office of Director of the Mint from April 1792 to June 1795, appointed by George Washington.
  Fact 1.1: Henry William de Saussure held the office from July 1795 to October 1795.
  Fact 1.2: Elias Boudinot held the office from October 1795 to July 1805.
  Fact 1.3: Robert Patterson held the office from January 1806 to July 1824, appointed by Thomas Jefferson.
  Fact 1.4: Samuel Moore held the office from July 1824 to July 1835, appointed by James Monroe.
  Fact 1.5: Robert M. Patterson held the office from May 1835 to July 1851, appointed by Andrew Jackson.
  Fact 1.6: George Nicholas Eckert held the office from July 1851 to April 1853, appointed by Millard Fillmore.
  Fact 1.7: Thomas M. Pettit held the office from April 1853 to May 1853, appointed by Franklin Pierce.
  Fact 1.8: James Ross Snowden held the office from June 1853 to May 1861.
  Fact 1.9: James Pollock held the office from May 1861 to September 1866, appointed by Abraham Lincoln.
  Fact 1.10: William Millward held the office from October 1866 to April 1867, appointed by Andrew Johnson.
  Fact 1.11: Henry Linderman held the office from April 1867 to May 1869.
  Fact 1.12: James Pollock held the office from May 1869 to March 1873, appointed by Ulysses S. Grant.
  Fact 1.13: Henry Linderman held the office from April 1873 to December 1878.
  Fact 1.14: Horatio C. Burchard held the office from February 1879 to June 1885, appointed by Rutherford B. Hayes.
  Fact 1.15: James P. Kimball held the office from July 1885 to October 1889, appointed by Grover Cleveland.
  Fact 1.16: Edward O. Leech held the office from October 1889 to May 1893, appointed by Benjamin Harrison.
  Fact 1.17: Robert E. Preston held the office from November 1893 to February 1898, appointed by Grover Cleveland.
  Fact 1.18: George E. Roberts held the office from February 1898 to July 1907, appointed by William McKinley.
  Fact 1.19: Frank A. Leach held the office from September 1907 to August 1909, appointed by Theodore Roosevelt.
  Fact 1.20: A. Piatt Andrew held the office from November 1909 to June 1910.
  Fact 1.21: George E. Roberts held the office from July 1910 to November 1914, appointed by William Howard Taft.
  Fact 1.22: Robert W. Woolley held the office from March 1915 to July 1916, appointed by Woodrow Wilson.
  Fact 1.23: F.H. von Engelken held the office from September 1916 to March 1917.
  Fact 1.24: Raymond T. Baker held the office from March 1917 to March 1922.
  Fact 1.25: Frank Edgar Scobey held the office from March 1922 to September 1923, appointed by Warren G. Harding.
  Fact 1.26: Robert J. Grant held the office from November 1923 to May 1933, appointed by Calvin Coolidge.
  Fact 1.27: Nellie Tayloe Ross held the office from May 1933 to April 1953, appointed by Franklin D. Roosevelt.
  Fact 1.28: William H. Brett held the office from July 1954 to January 1961, appointed by Dwight D. Eisenhower.
  Fact 1.29: Eva Adams held the office from October 1961 to August 1969, appointed by John F. Kennedy.
  Fact 1.30: Mary Brooks held the office from September 1969 to February 1977, appointed by Richard Nixon.
  Fact 1.31: Stella Hackel Sims held the office from November 1977 to April 1981, appointed by Jimmy Carter.
  Fact 1.32: Donna Pope held the office from July 1981 to August 1991, appointed by Ronald Reagan.
  Fact 1.33: David J. Ryder held the office from September 1992 to November 1993, appointed by George H.W. Bush.
  Fact 1.34: Philip N. Diehl held the office from June 1994 to March 2000, appointed by Bill Clinton.
  Fact 1.35: Jay W. Johnson held the office from May 2000 to August 2001.
  Fact 1.36: Henrietta H. Fore held the office from August 2001 to August 2005, appointed by George W. Bush.
  Fact 1.37: Edmund C. Moy held the office from September 2006 to January 2011.
  Fact 1.38: David J. Ryder has held the office since April 2018, appointed by Donald Trump.

Retrieved items:
<item 1>
Large denominations of United States currency
dollar U.S. Mint Philadelphia San Francisco Denver West Point Bureau of Engraving and Printing Mutilated currency Current coinage Penny ( 1 ¢ ) Nickel ( 5 ¢ )
Dime ( 10 ¢ ) Quarter ( 25 ¢ ) Half dollar ( 50 ¢ ) Dollar ( $1 ) Bullion coinage America the Beautiful Silver Bullion Coins American Buffalo American Gold
Eagle American Platinum Eagle American Silver Eagle American Palladium Eagle Current paper money $1 $2 $5 $10 $20 $50 $100 See also Bicentennial coinage
Commemoratives Early Commemoratives Modern Commemoratives Confederate dollar Large denominations Obsolete denominations Promotional fake denominations
Counterfeit United States currency Mill Coin production In God We Trust E pluribus unum Annuit cœptis Nicknames Replacement banknote Sales tax token ( hide )
Obsolete United States currency and coinage Topics United States coinage United States dollar History of the United States dollar Coins Half cent ( 1793 -- 1857
) Fugio Cent ( 1787 ) Large cent ( 1793 -- 1857 ;
</item 1>

<item 2>
United States two-dollar bill
dollar U.S. Mint Philadelphia San Francisco Denver West Point Bureau of Engraving and Printing Mutilated currency Current coinage Penny ( 1 ¢ ) Nickel ( 5 ¢ )
Dime ( 10 ¢ ) Quarter ( 25 ¢ ) Half dollar ( 50 ¢ ) Dollar ( $1 ) Bullion coinage America the Beautiful Silver Bullion Coins American Buffalo American Gold
Eagle American Platinum Eagle American Silver Eagle American Palladium Eagle Current paper money $1 $2 $5 $10 $20 $50 $100 See also Bicentennial coinage
Commemoratives Early Commemoratives Modern Commemoratives Confederate dollar Large denominations Obsolete denominations Promotional fake denominations
Counterfeit United States currency Mill Coin production In God We Trust E pluribus unum Annuit cœptis Nicknames Replacement banknote Sales tax token Retrieved
from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Director of the United States Mint
Director of the United States Mint - wikipedia Director of the United States Mint Jump to : navigation , search Seal of the United States Mint . The Director of
the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the
Mint has been Acting Principal Deputy Director David Motl . The office of Director has existed since the creation of the Mint by the Coinage Act of 1792 .
Initially appointed serving at the pleasure of the President of the United States , the Coinage Act of 1873 specified a five - year term for Directors . The
Director operates with general directions provided by the United States Secretary of the Treasury . Since the resignation of Edmund C. Moy in 2011 the mint has
been without an official director . Initially Matthew Rhett Jeppson was nominated by US President Barack Obama to become the mint 's 39th director and was given
the temporary title of principal deputy director , however the nomination was never confirmed by the Senate . Jeppson stepped down as principal deputy director
in January 2017 , being replaced by Acting Principal Deputy Director David Motl .
</item 3>

<item 4>
Director of the United States Mint
Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February
1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew
November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H.
</item 4>

<item 5>
Vice President of the United States
Vice president of the United States - wikipedia Vice president of the United States Jump to : navigation , search For a list of office - holders , see List of
Vice Presidents of the United States . Vice President of the United States of America Vice Presidential seal Vice Presidential standard Incumbent Mike Pence
since January 20 , 2017 Executive branch of the U.S. government Office of the Vice President Style Mr. Vice President ( Informal ) The Honorable ( Formal ) Mr.
President ( When presiding over Senate ) His Excellency ( In international correspondence ) Member of Cabinet National Security Council President of the U.S.
Senate Residence Number One Observatory Circle Seat Washington , D.C. Appointer Electoral College Term length 4 years Constituting instrument U.S.
</item 5>

<item 6>
Vice President of the United States
Senate vice presidential bust collection In fiction Presidents actors Vice Presidents actors Candidates Line of succession U.S. Government Portal Biography
Portal Presidents of the United States Presidential line of succession in the United States of America Vice President ( Mike Pence ) Speaker of the House of
Representatives ( Paul Ryan ) President pro tempore of the Senate ( Orrin Hatch ) Secretary of State ( Mike Pompeo ) Secretary of the Treasury ( Steven Mnuchin
) Secretary of Defense ( Jim Mattis ) Attorney General ( Jeff Sessions ) Secretary of the Interior ( Ryan Zinke ) Secretary of Agriculture ( Sonny Perdue )
Secretary of Commerce ( Wilbur Ross ) Secretary of Labor ( Alex Acosta ) Secretary of Health and Human Services ( Alex Azar ) Secretary of Housing and Urban
Development ( Ben Carson ) Secretary of Transportation ( Elaine Chao ) Secretary of Energy ( Rick Perry ) Secretary of Education ( Betsy DeVos ) Secretary of
Veterans Affairs ( Robert Wilkie ) Secretary of Homeland Security ( Kirstjen Nielsen ) Jump up ^ Although Elaine Chao is the current Secretary of Transportation
, she is not a natural - born citizen ( acquired U.S. citizenship by naturalization ) and thus ineligible for the presidency .
</item 6>

<item 7>
Director of the United States Mint
US Mint . Retrieved 13 April 2017 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Director_of_the_United_States_Mint&oldid=836791366 ''
Categories : Directors of the United States Mint United States Department of the Treasury 1792 establishments in the United States Talk About Wikipedia Add
links This page was last edited on 16 April 2018 , at 21 : 26 . About Wikipedia
</item 7>

<item 8>
Director of the United States Mint
List of Directors of the United States Mint ( edit ) Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George
Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson
Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore
Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William
Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April
1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O.
</item 8>

<item 9>
Director of the United States Mint
Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C.
Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump References ( edit ) Jump up ^ About the Mint at usmint.gov ( retrieved 20 April
2015 ) . Jump up ^ Gilkes , Paul ( 23 January 2017 ) . `` Rhett Jeppson leaves U.S. Mint as principal deputy director '' . coinworld.com . Retrieved 13 April
2017 . Jump up ^ `` Director 's Office '' . US Mint . Retrieved 13 April 2017 .
</item 9>

<item 10>
Director of the United States Mint
Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922
September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H.
Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon
Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W.
</item 10>
=== END ITEM id=a2a622b1158dd55c ===

=== ITEM id=b46555a209fe2d2f ===
Gold sentences and facts:
Sentence 0: Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France `` for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy '' Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the properties of matter at low temperatures which led , inter alia , to the production of liquid helium '' 1914 Max von Laue Germany `` For his discovery of the diffraction of X-rays by crystals '' , an important step in the development of X-ray spectroscopy . 1915 William Henry Bragg United Kingdom `` For their services in the analysis of crystal structure by means of X-rays '' , an important step in the development of X-ray crystallography William Lawrence Bragg Australia United Kingdom 1916 Not awarded World War I 1917 Charles Glover Barkla United Kingdom `` For his discovery of the characteristic Röntgen radiation of the elements '' , another important step in the development of X-ray spectroscopy 1918 Max Planck Germany `` for the services he rendered to the advancement of physics by his discovery of energy quanta '' 1919 Johannes Stark Germany `` for his discovery of the Doppler effect in canal rays and the splitting of spectral lines in electric fields '' 1920 Charles Édouard Guillaume Switzerland `` for the service he has rendered to precision measurements in physics by his discovery of anomalies in nickel - steel alloys '' 1921 Albert Einstein Germany Switzerland `` for his services to theoretical physics , and especially for his discovery of the law of the photoelectric effect '' 1922 Niels Bohr Denmark `` for his services in the investigation of the structure of atoms and of the radiation emanating from them '' 1923 Robert Andrews Millikan United States `` for his work on the elementary charge of electricity and on the photoelectric effect '' 1924 Manne Siegbahn Sweden `` for his discoveries and research in the field of X-ray spectroscopy '' 1925 James Franck Germany `` for their discovery of the laws governing the impact of an electron upon an atom '' Gustav Hertz Germany 1926 Jean Baptiste Perrin France `` for his work on the discontinuous structure of matter , and especially for his discovery of sedimentation equilibrium '' 1927 Arthur Holly Compton United States `` for his discovery of the effect named after him '' Charles Thomson Rees Wilson United Kingdom `` for his method of making the paths of electrically charged particles visible by condensation of vapour '' 1928 Owen Willans Richardson United Kingdom `` for his work on the thermionic phenomenon and especially for the discovery of the law named after him '' 1929 Louis Victor Pierre Raymond , 7th Duc de Broglie France `` for his discovery of the wave nature of electrons '' 1930 Chandrasekhara Venkata Raman India `` for his work on the scattering of light and for the discovery of the effect named after him '' 1931 Not awarded 1932 Werner Heisenberg Germany `` for the creation of quantum mechanics , the application of which has , inter alia , led to the discovery of the allotropic forms of hydrogen '' 1933 Erwin Schrödinger Austria `` for the discovery of new productive forms of atomic theory '' Paul Dirac United Kingdom 1934 Not awarded 1935 James Chadwick United Kingdom `` for the discovery of the neutron '' 1936 Victor Francis Hess Austria `` for his discovery of cosmic radiation '' Carl David Anderson United States `` for his discovery of the positron '' 1937 Clinton Joseph Davisson United States `` for their experimental discovery of the diffraction of electrons by crystals '' George Paget Thomson United Kingdom 1938 Enrico Fermi Italy `` for his demonstrations of the existence of new radioactive elements produced by neutron irradiation , and for his related discovery of nuclear reactions brought about by slow neutrons '' 1939 Ernest Lawrence United States `` for the invention and development of the cyclotron and for results obtained with it , especially with regard to artificial radioactive elements '' 1940 Not awarded World War II 1941 Not awarded World War II 1942 Not awarded World War II 1943 Otto Stern United States `` for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton '' 1944 Isidor Isaac Rabi United States Poland `` for his resonance method for recording the magnetic properties of atomic nuclei '' 1945 Wolfgang Pauli Austria `` for the discovery of the Exclusion Principle , also called the Pauli principle '' 1946 Percy Williams Bridgman United States `` for the invention of an apparatus to produce extremely high pressures , and for the discoveries he made there within the field of high pressure physics '' 1947 Edward Victor Appleton United Kingdom `` for his investigations of the physics of the upper atmosphere especially for the discovery of the so - called Appleton layer '' 1948 Patrick Maynard Stuart Blackett United Kingdom `` for his development of the Wilson cloud chamber method , and his discoveries therewith in the fields of nuclear physics and cosmic radiation '' 1949 Hideki Yukawa Japan `` for his prediction of the existence of mesons on the basis of theoretical work on nuclear forces '' 1950 Cecil Frank Powell United Kingdom `` for his development of the photographic method of studying nuclear processes and his discoveries regarding mesons made with this method '' 1951 John Douglas Cockcroft United Kingdom `` for their pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles '' Ernest Thomas Sinton Walton Ireland 1952 Felix Bloch Switzerland United States `` for their development of new methods for nuclear magnetic precision measurements and discoveries in connection therewith '' Edward Mills Purcell United States 1953 Frits Zernike Netherlands `` for his demonstration of the phase contrast method , especially for his invention of the phase contrast microscope '' 1954 Max Born West Germany `` for his fundamental research in quantum mechanics , especially for his statistical interpretation of the wavefunction '' Walther Bothe West Germany `` for the coincidence method and his discoveries made therewith '' 1955 Willis Eugene Lamb United States `` for his discoveries concerning the fine structure of the hydrogen spectrum '' Polykarp Kusch United States Germany `` for his precision determination of the magnetic moment of the electron '' 1956 John Bardeen United States `` for their researches on semiconductors and their discovery of the transistor effect '' Walter Houser Brattain United States William Bradford Shockley United States 1957 Tsung - Dao Lee Republic of China `` for their penetrating investigation of the so - called parity laws which has led to important discoveries regarding the elementary particles '' Chen Ning Yang Republic of China 1958 Pavel Alekseyevich Cherenkov Soviet Union `` for the discovery and the interpretation of the Cherenkov effect '' Ilya Frank Soviet Union Igor Yevgenyevich Tamm Soviet Union 1959 Emilio Gino Segrè Italy United States `` for their discovery of the antiproton '' Owen Chamberlain United States 1960 Donald Arthur Glaser United States `` for the invention of the bubble chamber '' 1961 Robert Hofstadter United States `` for his pioneering studies of electron scattering in atomic nuclei and for his thereby achieved discoveries concerning the structure of the nucleons '' Rudolf Ludwig Mössbauer West Germany `` for his researches concerning the resonance absorption of gamma radiation and his discovery in this connection of the effect which bears his name '' 1962 Lev Davidovich Landau Soviet Union `` for his pioneering theories for condensed matter , especially liquid helium '' 1963 Eugene Paul Wigner Hungary United States `` for his contributions to the theory of the atomic nucleus and the elementary particles , particularly through the discovery and application of fundamental symmetry principles '' Maria Goeppert - Mayer United States `` for their discoveries concerning nuclear shell structure '' J. Hans D. Jensen West Germany 1964 Nicolay Gennadiyevich Basov Soviet Union `` for fundamental work in the field of quantum electronics , which has led to the construction of oscillators and amplifiers based on the maser -- laser principle '' Alexander Prokhorov Soviet Union Charles Hard Townes United States 1965 Richard Phillips Feynman United States `` for their fundamental work in quantum electrodynamics ( QED ) , with deep - ploughing consequences for the physics of elementary particles '' Julian Schwinger United States Shin'ichirō Tomonaga Japan 1966 Alfred Kastler France `` for the discovery and development of optical methods for studying Hertzian resonances in atoms '' 1967 Hans Albrecht Bethe United States `` for his contributions to the theory of nuclear reactions , especially his discoveries concerning the energy production in stars '' 1968 Luis Walter Alvarez United States `` for his decisive contributions to elementary particle physics , in particular the discovery of a large number of resonance states , made possible through his development of the technique of using hydrogen bubble chamber and data analysis '' 1969 Murray Gell - Mann United States `` for his contributions and discoveries concerning the classification of elementary particles and their interactions '' 1970 Hannes Olof Gösta Alfvén Sweden `` for fundamental work and discoveries in magneto - hydrodynamics with fruitful applications in different parts of plasma physics '' Louis Néel France `` for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism which have led to important applications in solid state physics '' 1971 Dennis Gabor Hungary United Kingdom `` for his invention and development of the holographic method '' 1972 John Bardeen United States `` for their jointly developed theory of superconductivity , usually called the BCS - theory '' Leon Neil Cooper United States John Robert Schrieffer United States 1973 Leo Esaki Japan `` for their experimental discoveries regarding tunneling phenomena in semiconductors and superconductors , respectively '' Ivar Giaever United States Norway Brian David Josephson United Kingdom `` for his theoretical predictions of the properties of a supercurrent through a tunnel barrier , in particular those phenomena which are generally known as the Josephson effect '' Martin Ryle United Kingdom `` for their pioneering research in radio astrophysics : Ryle for his observations and inventions , in particular of the aperture synthesis technique , and Hewish for his decisive role in the discovery of pulsars '' Antony Hewish United Kingdom Aage Bohr Denmark `` for the discovery of the connection between collective motion and particle motion in atomic nuclei and the development of the theory of the structure of the atomic nucleus based on this connection '' Ben Roy Mottelson Denmark Leo James Rainwater United States 1976 Burton Richter United States `` for their pioneering work in the discovery of a heavy elementary particle of a new kind '' Samuel Chao Chung Ting United States 1977 Philip Warren Anderson United States `` for their fundamental theoretical investigations of the electronic structure of magnetic and disordered systems '' Nevill Francis Mott United Kingdom John Hasbrouck Van Vleck United States 1978 Pyotr Leonidovich Kapitsa Soviet Union `` for his basic inventions and discoveries in the area of low - temperature physics '' Arno Allan Penzias United States `` for their discovery of cosmic microwave background radiation '' Robert Woodrow Wilson United States 1979 Sheldon Lee Glashow United States `` for their contributions to the theory of the unified weak and electromagnetic interaction between elementary particles , including , inter alia , the prediction of the weak neutral current '' Abdus Salam Pakistan Steven Weinberg United States 1980 James Watson Cronin United States `` for the discovery of violations of fundamental symmetry principles in the decay of neutral K - mesons '' Val Logsdon Fitch United States 1981 Nicolaas Bloembergen Netherlands United States `` for their contribution to the development of laser spectroscopy '' Arthur Leonard Schawlow United States Kai Manne Börje Siegbahn Sweden `` for his contribution to the development of high - resolution electron spectroscopy '' 1982 Kenneth G. Wilson United States `` for his theory for critical phenomena in connection with phase transitions '' Subrahmanyan Chandrasekhar India United States `` for his theoretical studies of the physical processes of importance to the structure and evolution of the stars '' William Alfred Fowler United States `` for his theoretical and experimental studies of the nuclear reactions of importance in the formation of the chemical elements in the universe '' 1984 Carlo Rubbia Italy `` for their decisive contributions to the large project , which led to the discovery of the field particles W and Z , communicators of weak interaction '' Simon van der Meer Netherlands 1985 Klaus von Klitzing West Germany `` for the discovery of the quantized Hall effect '' 1986 Ernst Ruska West Germany `` for his fundamental work in electron optics , and for the design of the first electron microscope '' Gerd Binnig West Germany `` for their design of the scanning tunneling microscope '' Heinrich Rohrer Switzerland Johannes Georg Bednorz West Germany `` for their important break - through in the discovery of superconductivity in ceramic materials '' Karl Alexander Müller Switzerland 1988 Leon Max Lederman United States `` for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino '' Melvin Schwartz United States Jack Steinberger United States 1989 Norman Foster Ramsey United States `` for the invention of the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks '' Hans Georg Dehmelt United States Germany `` for the development of the ion trap technique '' Wolfgang Paul West Germany 1990 Jerome I. Friedman United States `` for their pioneering investigations concerning deep inelastic scattering of electrons on protons and bound neutrons , which have been of essential importance for the development of the quark model in particle physics '' Henry Way Kendall United States Richard E. Taylor Canada 1991 Pierre - Gilles de Gennes France `` for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter , in particular to liquid crystals and polymers '' 1992 Georges Charpak France Poland `` for his invention and development of particle detectors , in particular the multiwire proportional chamber '' 1993 Russell Alan Hulse United States `` for the discovery of a new type of pulsar , a discovery that has opened up new possibilities for the study of gravitation '' Joseph Hooton Taylor , Jr . United States 1994 Bertram Brockhouse Canada `` for the development of neutron spectroscopy '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' Clifford Glenwood Shull United States `` for the development of the neutron diffraction technique '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' 1995 Martin Lewis Perl United States `` for the discovery of the tau lepton '' and `` for pioneering experimental contributions to lepton physics '' Frederick Reines United States `` for the detection of the neutrino '' and `` for pioneering experimental contributions to lepton physics '' David Morris Lee United States `` for their discovery of superfluidity in helium - 3 '' Douglas D. Osheroff United States Robert Coleman Richardson United States 1997 Steven Chu United States `` for development of methods to cool and trap atoms with laser light . '' Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands `` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002 Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics , in particular for the detection of cosmic neutrinos '' Masatoshi Koshiba Japan Riccardo Giacconi Italy United States `` for pioneering contributions to astrophysics , which have led to the discovery of cosmic X-ray sources '' 2003 Alexei Alexeyevich Abrikosov Russia United States `` for pioneering contributions to the theory of superconductors and superfluids '' Vitaly Lazarevich Ginzburg Russia Anthony James Leggett United Kingdom United States David J. Gross United States `` for the discovery of asymptotic freedom in the theory of the strong interaction '' Hugh David Politzer United States Frank Wilczek United States 2005 Roy J. Glauber United States `` for his contribution to the quantum theory of optical coherence '' John L. Hall United States `` for their contributions to the development of laser - based precision spectroscopy , including the optical frequency comb technique '' Theodor W. Hänsch Germany 2006 John C. Mather United States `` for their discovery of the blackbody form and anisotropy of the cosmic microwave background radiation '' George F. Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and manipulation of individual quantum systems . '' David J. Wineland United States 2013 François Englert Belgium `` for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles , and which recently was confirmed through the discovery of the predicted fundamental particle , by the ATLAS and CMS experiments at CERN 's Large Hadron Collider '' Peter Higgs United Kingdom 2014 Isamu Akasaki Japan `` for the invention of efficient blue light - emitting diodes which has enabled bright and energy - saving white light sources '' Hiroshi Amano Japan Shuji Nakamura Japan United States 2015 Takaaki Kajita Japan `` for the discovery of neutrino oscillations , which shows that neutrinos have mass '' Arthur B. McDonald Canada 2016 David J. Thouless United Kingdom United States `` for theoretical discoveries of topological phase transitions and topological phases of matter '' F. Duncan M. Haldane United Kingdom United States John M. Kosterlitz United Kingdom United States 2017 Rainer Weiss Germany United States `` for decisive contributions to the LIGO detector and the observation of gravitational waves '' Kip Thorne United States Barry Barish United States
  Fact 0.0: 1901: Wilhelm Conrad Röntgen of Germany won the physics prize for the discovery of the remarkable rays later named after him
  Fact 0.1: 1902: Hendrik Lorentz of the Netherlands won the physics prize for research into the influence of magnetism upon radiation phenomena
  Fact 0.2: 1902: Pieter Zeeman of the Netherlands shared the physics prize with Hendrik Lorentz
  Fact 0.3: 1903: Antoine Henri Becquerel of France won the physics prize for the discovery of spontaneous radioactivity
  Fact 0.4: 1903: Pierre Curie of France shared the physics prize for joint research on the radiation phenomena discovered by Henri Becquerel
  Fact 0.5: 1903: Maria Skłodowska-Curie, listed as Poland and France, shared the physics prize for the same joint research on radiation phenomena
  Fact 0.6: 1904: Lord Rayleigh of the United Kingdom won the physics prize for investigating the densities of the most important gases
  Fact 0.7: 1904: Lord Rayleigh of the United Kingdom won the physics prize for discovering argon
  Fact 0.8: 1905: Philipp Eduard Anton von Lenard, listed as Austria-Hungary and Germany, won the physics prize for his work on cathode rays
  Fact 0.9: 1906: Joseph John Thomson of the United Kingdom won the physics prize for theoretical and experimental investigations of electricity conduction by gases
  Fact 0.10: 1907: Albert Abraham Michelson, listed as the United States and Poland, won the physics prize for optical precision instruments
  Fact 0.11: 1907: Albert Abraham Michelson won the physics prize for spectroscopic and metrological investigations carried out with those instruments
  Fact 0.12: 1908: Gabriel Lippmann of France won the physics prize for a method of reproducing colours photographically based on interference
  Fact 0.13: 1909: Guglielmo Marconi of Italy won the physics prize for contributions to the development of wireless telegraphy
  Fact 0.14: 1909: Karl Ferdinand Braun of Germany shared the physics prize for contributions to wireless telegraphy
  Fact 0.15: 1910: Johannes Diderik van der Waals of the Netherlands won the physics prize for work on the equation of state for gases and liquids
  Fact 0.16: 1911: Wilhelm Wien of Germany won the physics prize for discoveries regarding the laws governing the radiation of heat
  Fact 0.17: 1912: Nils Gustaf Dalén of Sweden won the physics prize for inventing automatic valves used with gas accumulators in lighthouses and buoys
  Fact 0.18: 1913: Heike Kamerlingh-Onnes of the Netherlands won the physics prize for investigations of matter at low temperatures, which led to the production of liquid helium
  Fact 0.19: 1914: Max von Laue of Germany won the physics prize for discovering the diffraction of X-rays by crystals
  Fact 0.20: 1915: William Henry Bragg of the United Kingdom shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.21: 1915: William Lawrence Bragg, listed as Australia and the United Kingdom, shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.22: 1916: No physics prize was awarded because of World War I
  Fact 0.23: 1917: Charles Glover Barkla of the United Kingdom won the physics prize for discovering the characteristic Röntgen radiation of the elements
  Fact 0.24: 1918: Max Planck of Germany won the physics prize for services to physics through his discovery of energy quanta
  Fact 0.25: 1919: Johannes Stark of Germany won the physics prize for discovering the Doppler effect in canal rays and the splitting of spectral lines in electric fields
  Fact 0.26: 1920: Charles Édouard Guillaume of Switzerland won the physics prize for precision measurements in physics through his discovery of anomalies in nickel-steel alloys
  Fact 0.27: 1921: Albert Einstein, listed as Germany and Switzerland, won the physics prize for services to theoretical physics
  Fact 0.28: 1921: Albert Einstein won the physics prize especially for discovering the law of the photoelectric effect
  Fact 0.29: 1922: Niels Bohr of Denmark won the physics prize for investigating the structure of atoms and the radiation emanating from them
  Fact 0.30: 1923: Robert Andrews Millikan of the United States won the physics prize for work on the elementary charge of electricity and on the photoelectric effect
  Fact 0.31: 1924: Manne Siegbahn of Sweden won the physics prize for discoveries and research in X-ray spectroscopy
  Fact 0.32: 1925: James Franck of Germany won the physics prize for discovering the laws governing the impact of an electron upon an atom
  Fact 0.33: 1925: Gustav Hertz of Germany shared the physics prize with James Franck
  Fact 0.34: 1926: Jean Baptiste Perrin of France won the physics prize for work on the discontinuous structure of matter, especially discovering sedimentation equilibrium
  Fact 0.35: 1927: Arthur Holly Compton of the United States won the physics prize for discovering the effect named after him
  Fact 0.36: 1927: Charles Thomson Rees Wilson of the United Kingdom shared the physics prize for a method of making charged particle paths visible by condensation of vapour
  Fact 0.37: 1928: Owen Willans Richardson of the United Kingdom won the physics prize for work on the thermionic phenomenon and discovering the law named after him
  Fact 0.38: 1929: Louis Victor Pierre Raymond, 7th Duc de Broglie, of France won the physics prize for discovering the wave nature of electrons
  Fact 0.39: 1930: Chandrasekhara Venkata Raman of India won the physics prize for work on the scattering of light and discovering the effect named after him
  Fact 0.40: 1931: No physics prize was awarded
  Fact 0.41: 1932: Werner Heisenberg of Germany won the physics prize for creating quantum mechanics, whose application led to the discovery of allotropic forms of hydrogen
  Fact 0.42: 1933: Erwin Schrödinger of Austria won the physics prize for discovering new productive forms of atomic theory
  Fact 0.43: 1933: Paul Dirac of the United Kingdom shared the physics prize for work on atomic theory
  Fact 0.44: 1934: No physics prize was awarded
  Fact 0.45: 1935: James Chadwick of the United Kingdom won the physics prize for discovering the neutron
  Fact 0.46: 1936: Victor Francis Hess of Austria won the physics prize for discovering cosmic radiation
  Fact 0.47: 1936: Carl David Anderson of the United States shared the physics prize for discovering the positron
  Fact 0.48: 1937: Clinton Joseph Davisson of the United States shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.49: 1937: George Paget Thomson of the United Kingdom shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.50: 1938: Enrico Fermi of Italy won the physics prize for demonstrating new radioactive elements produced by neutron irradiation
  Fact 0.51: 1938: Enrico Fermi won the physics prize for discovering nuclear reactions brought about by slow neutrons
  Fact 0.52: 1939: Ernest Lawrence of the United States won the physics prize for inventing and developing the cyclotron
  Fact 0.53: 1939: Ernest Lawrence won the physics prize for results obtained with the cyclotron, especially on artificial radioactive elements
  Fact 0.54: 1940: No physics prize was awarded because of World War II
  Fact 0.55: 1941: No physics prize was awarded because of World War II
  Fact 0.56: 1942: No physics prize was awarded because of World War II
  Fact 0.57: 1943: Otto Stern of the United States won the physics prize for developing the molecular ray method
  Fact 0.58: 1943: Otto Stern won the physics prize for discovering the magnetic moment of the proton
  Fact 0.59: 1944: Isidor Isaac Rabi, listed as the United States and Poland, won the physics prize for his resonance method for recording the magnetic properties of atomic nuclei
  Fact 0.60: 1945: Wolfgang Pauli of Austria won the physics prize for discovering the Exclusion Principle, also called the Pauli principle
  Fact 0.61: 1946: Percy Williams Bridgman of the United States won the physics prize for inventing an apparatus to produce extremely high pressures
  Fact 0.62: 1946: Percy Williams Bridgman won the physics prize for discoveries he made within high pressure physics
  Fact 0.63: 1947: Edward Victor Appleton of the United Kingdom won the physics prize for investigating the physics of the upper atmosphere
  Fact 0.64: 1947: Edward Victor Appleton won the physics prize especially for discovering the so-called Appleton layer
  Fact 0.65: 1948: Patrick Maynard Stuart Blackett of the United Kingdom won the physics prize for developing the Wilson cloud chamber method
  Fact 0.66: 1948: Patrick Maynard Stuart Blackett won the physics prize for discoveries with that method in nuclear physics and cosmic radiation
  Fact 0.67: 1949: Hideki Yukawa of Japan won the physics prize for predicting the existence of mesons from theoretical work on nuclear forces
  Fact 0.68: 1950: Cecil Frank Powell of the United Kingdom won the physics prize for developing the photographic method of studying nuclear processes
  Fact 0.69: 1950: Cecil Frank Powell won the physics prize for discoveries regarding mesons made with that method
  Fact 0.70: 1951: John Douglas Cockcroft of the United Kingdom shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.71: 1951: Ernest Thomas Sinton Walton of Ireland shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.72: 1952: Felix Bloch, listed as Switzerland and the United States, shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.73: 1952: Edward Mills Purcell of the United States shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.74: 1953: Frits Zernike of the Netherlands won the physics prize for demonstrating the phase contrast method
  Fact 0.75: 1953: Frits Zernike won the physics prize especially for inventing the phase contrast microscope
  Fact 0.76: 1954: Max Born of West Germany won the physics prize for fundamental research in quantum mechanics
  Fact 0.77: 1954: Max Born won the physics prize especially for his statistical interpretation of the wavefunction
  Fact 0.78: 1954: Walther Bothe of West Germany shared the physics prize for the coincidence method and his discoveries made with it
  Fact 0.79: 1955: Willis Eugene Lamb of the United States won the physics prize for discoveries concerning the fine structure of the hydrogen spectrum
  Fact 0.80: 1955: Polykarp Kusch, listed as the United States and Germany, shared the physics prize for precisely determining the magnetic moment of the electron
  Fact 0.81: 1956: John Bardeen of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.82: 1956: Walter Houser Brattain of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.83: 1956: William Bradford Shockley of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.84: 1957: Tsung-Dao Lee of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.85: 1957: Chen Ning Yang of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.86: 1958: Pavel Alekseyevich Cherenkov of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.87: 1958: Ilya Frank of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.88: 1958: Igor Yevgenyevich Tamm of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.89: 1959: Emilio Gino Segrè, listed as Italy and the United States, shared the physics prize for discovering the antiproton
  Fact 0.90: 1959: Owen Chamberlain of the United States shared the physics prize for discovering the antiproton
  Fact 0.91: 1960: Donald Arthur Glaser of the United States won the physics prize for inventing the bubble chamber
  Fact 0.92: 1961: Robert Hofstadter of the United States shared the physics prize for pioneering studies of electron scattering in atomic nuclei
  Fact 0.93: 1961: Robert Hofstadter shared the physics prize for discoveries about the structure of the nucleons
  Fact 0.94: 1961: Rudolf Ludwig Mössbauer of West Germany shared the physics prize for researches on the resonance absorption of gamma radiation
  Fact 0.95: 1961: Rudolf Ludwig Mössbauer shared the physics prize for discovering the effect that bears his name
  Fact 0.96: 1962: Lev Davidovich Landau of the Soviet Union won the physics prize for pioneering theories of condensed matter, especially liquid helium
  Fact 0.97: 1963: Eugene Paul Wigner, listed as Hungary and the United States, shared the physics prize for contributions to the theory of the atomic nucleus and elementary particles
  Fact 0.98: 1963: Eugene Paul Wigner shared the physics prize for discovering and applying fundamental symmetry principles
  Fact 0.99: 1963: Maria Goeppert-Mayer of the United States shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.100: 1963: J. Hans D. Jensen of West Germany shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.101: 1964: Nicolay Gennadiyevich Basov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.102: 1964: Alexander Prokhorov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.103: 1964: Charles Hard Townes of the United States shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.104: 1965: Richard Phillips Feynman of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.105: 1965: Julian Schwinger of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.106: 1965: Shin'ichirō Tomonaga of Japan shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.107: 1966: Alfred Kastler of France won the physics prize for discovering and developing optical methods for studying Hertzian resonances in atoms
  Fact 0.108: 1967: Hans Albrecht Bethe of the United States won the physics prize for contributions to the theory of nuclear reactions
  Fact 0.109: 1967: Hans Albrecht Bethe won the physics prize especially for discoveries about energy production in stars
  Fact 0.110: 1968: Luis Walter Alvarez of the United States won the physics prize for decisive contributions to elementary particle physics
  Fact 0.111: 1968: Luis Walter Alvarez won the physics prize especially for discovering many resonance states using the hydrogen bubble chamber technique
  Fact 0.112: 1969: Murray Gell-Mann of the United States won the physics prize for contributions and discoveries concerning the classification of elementary particles and their interactions
  Fact 0.113: 1970: Hannes Olof Gösta Alfvén of Sweden shared the physics prize for fundamental work in magneto-hydrodynamics
  Fact 0.114: 1970: Louis Néel of France shared the physics prize for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism
  Fact 0.115: 1971: Dennis Gabor, listed as Hungary and the United Kingdom, won the physics prize for inventing and developing the holographic method
  Fact 0.116: 1972: John Bardeen of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.117: 1972: Leon Neil Cooper of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.118: 1972: John Robert Schrieffer of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.119: 1973: Leo Esaki of Japan shared the physics prize for experimental discoveries regarding tunneling phenomena in semiconductors
  Fact 0.120: 1973: Ivar Giaever, listed as the United States and Norway, shared the physics prize for experimental discoveries regarding tunneling phenomena in superconductors
  Fact 0.121: 1973: Brian David Josephson of the United Kingdom shared the physics prize for theoretical predictions of supercurrent properties through a tunnel barrier
  Fact 0.122: 1973: Brian David Josephson shared the physics prize for predicting the phenomena known as the Josephson effect
  Fact 0.123: 1974: Martin Ryle of the United Kingdom shared the physics prize for pioneering research in radio astrophysics
  Fact 0.124: 1974: Martin Ryle shared the physics prize for inventing the aperture synthesis technique
  Fact 0.125: 1974: Antony Hewish of the United Kingdom shared the physics prize for his decisive role in discovering pulsars
  Fact 0.126: 1975: Aage Bohr of Denmark shared the physics prize for discovering the connection between collective motion and particle motion in atomic nuclei
  Fact 0.127: 1975: Ben Roy Mottelson of Denmark shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.128: 1975: Leo James Rainwater of the United States shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.129: 1976: Burton Richter of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.130: 1976: Samuel Chao Chung Ting of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.131: 1977: Philip Warren Anderson of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.132: 1977: Nevill Francis Mott of the United Kingdom shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.133: 1977: John Hasbrouck Van Vleck of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.134: 1978: Pyotr Leonidovich Kapitsa of the Soviet Union shared the physics prize for basic inventions and discoveries in low-temperature physics
  Fact 0.135: 1978: Arno Allan Penzias of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.136: 1978: Robert Woodrow Wilson of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.137: 1979: Sheldon Lee Glashow of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.138: 1979: Sheldon Lee Glashow shared the physics prize for predicting the weak neutral current
  Fact 0.139: 1979: Abdus Salam of Pakistan shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.140: 1979: Steven Weinberg of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.141: 1980: James Watson Cronin of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.142: 1980: Val Logsdon Fitch of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.143: 1981: Nicolaas Bloembergen, listed as the Netherlands and the United States, shared the physics prize for contributions to laser spectroscopy
  Fact 0.144: 1981: Arthur Leonard Schawlow of the United States shared the physics prize for contributions to laser spectroscopy
  Fact 0.145: 1981: Kai Manne Börje Siegbahn of Sweden shared the physics prize for contributions to high-resolution electron spectroscopy
  Fact 0.146: 1982: Kenneth G. Wilson of the United States won the physics prize for his theory of critical phenomena in connection with phase transitions
  Fact 0.147: 1982: Subrahmanyan Chandrasekhar, listed as India and the United States, shared the physics prize for theoretical studies of physical processes important to the structure and evolution of stars
  Fact 0.148: 1982: William Alfred Fowler of the United States shared the physics prize for theoretical and experimental studies of nuclear reactions important in forming the chemical elements
  Fact 0.149: 1984: Carlo Rubbia of Italy shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.150: 1984: Simon van der Meer of the Netherlands shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.151: 1985: Klaus von Klitzing of West Germany won the physics prize for discovering the quantized Hall effect
  Fact 0.152: 1986: Ernst Ruska of West Germany shared the physics prize for fundamental work in electron optics
  Fact 0.153: 1986: Ernst Ruska shared the physics prize for designing the first electron microscope
  Fact 0.154: 1986: Gerd Binnig of West Germany shared the physics prize for designing the scanning tunneling microscope
  Fact 0.155: 1986: Heinrich Rohrer of Switzerland shared the physics prize for designing the scanning tunneling microscope
  Fact 0.156: 1986: Johannes Georg Bednorz of West Germany shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.157: 1986: Karl Alexander Müller of Switzerland shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.158: 1988: Leon Max Lederman of the United States shared the physics prize for the neutrino beam method
  Fact 0.159: 1988: Leon Max Lederman shared the physics prize for demonstrating the doublet structure of the leptons through discovering the muon neutrino
  Fact 0.160: 1988: Melvin Schwartz of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.161: 1988: Jack Steinberger of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.162: 1989: Norman Foster Ramsey of the United States shared the physics prize for inventing the separated oscillatory fields method
  Fact 0.163: 1989: Norman Foster Ramsey shared the physics prize for its use in the hydrogen maser and other atomic clocks
  Fact 0.164: 1989: Hans Georg Dehmelt, listed as the United States and Germany, shared the physics prize for developing the ion trap technique
  Fact 0.165: 1989: Wolfgang Paul of West Germany shared the physics prize for developing the ion trap technique
  Fact 0.166: 1990: Jerome I. Friedman of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.167: 1990: Henry Way Kendall of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.168: 1990: Richard E. Taylor of Canada shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.169: 1991: Pierre-Gilles de Gennes of France won the physics prize for discovering that methods for order phenomena in simple systems generalize to liquid crystals and polymers
  Fact 0.170: 1992: Georges Charpak, listed as France and Poland, won the physics prize for inventing and developing particle detectors
  Fact 0.171: 1992: Georges Charpak won the physics prize especially for the multiwire proportional chamber
  Fact 0.172: 1993: Russell Alan Hulse of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.173: 1993: Joseph Hooton Taylor Jr. of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.174: 1994: Bertram Brockhouse of Canada shared the physics prize for developing neutron spectroscopy
  Fact 0.175: 1994: Bertram Brockhouse shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.176: 1994: Clifford Glenwood Shull of the United States shared the physics prize for developing the neutron diffraction technique
  Fact 0.177: 1994: Clifford Glenwood Shull shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.178: 1995: Martin Lewis Perl of the United States shared the physics prize for discovering the tau lepton
  Fact 0.179: 1995: Martin Lewis Perl shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.180: 1995: Frederick Reines of the United States shared the physics prize for detecting the neutrino
  Fact 0.181: 1995: Frederick Reines shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.182: 1995: David Morris Lee of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.183: 1995: Douglas D. Osheroff of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.184: 1995: Robert Coleman Richardson of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.185: 1997: Steven Chu of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.186: 1997: Claude Cohen-Tannoudji of France shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.187: 1997: William Daniel Phillips of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.188: 1998: Robert B. Laughlin of the United States shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.189: 1998: Horst Ludwig Störmer of Germany shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.190: 1998: Daniel Chee Tsui, listed as the Republic of China and the United States, shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.191: 1999: Gerard 't Hooft of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.192: 1999: Martinus J.G. Veltman of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.193: 2000: Zhores Ivanovich Alferov of Russia shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.194: 2000: Herbert Kroemer of Germany shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.195: 2000: Jack St. Clair Kilby of the United States shared the physics prize for his part in the invention of the integrated circuit
  Fact 0.196: 2001: Eric Allin Cornell of the United States shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.197: 2001: Carl Edwin Wieman of the United States shared the physics prize for early fundamental studies of the properties of the condensates
  Fact 0.198: 2001: Wolfgang Ketterle of Germany shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.199: 2002: Raymond Davis Jr. of the United States shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.200: 2002: Masatoshi Koshiba of Japan shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.201: 2002: Riccardo Giacconi, listed as Italy and the United States, shared the physics prize for pioneering contributions to astrophysics that led to the discovery of cosmic X-ray sources
  Fact 0.202: 2003: Alexei Alexeyevich Abrikosov, listed as Russia and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.203: 2003: Vitaly Lazarevich Ginzburg of Russia shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.204: 2003: Anthony James Leggett, listed as the United Kingdom and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.205: 2003: David J. Gross of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.206: 2003: Hugh David Politzer of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.207: 2003: Frank Wilczek of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.208: 2005: Roy J. Glauber of the United States won the physics prize for his contribution to the quantum theory of optical coherence
  Fact 0.209: 2005: John L. Hall of the United States shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.210: 2005: Theodor W. Hänsch of Germany shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.211: 2006: John C. Mather of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.212: 2006: John C. Mather shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.213: 2006: George F. Smoot of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.214: 2006: George F. Smoot shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.215: 2007: Albert Fert of France shared the physics prize for discovering giant magnetoresistance
  Fact 0.216: 2007: Peter Grünberg of Germany shared the physics prize for discovering giant magnetoresistance
  Fact 0.217: 2008: Makoto Kobayashi of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.218: 2008: Toshihide Maskawa of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.219: 2008: Yoichiro Nambu, listed as Japan and the United States, shared the physics prize for discovering the mechanism of spontaneous broken symmetry in subatomic physics
  Fact 0.220: 2009: Charles K. Kao, listed as Hong Kong, the United Kingdom and the United States, shared the physics prize for groundbreaking achievements concerning light transmission in fibers for optical communication
  Fact 0.221: 2009: Willard S. Boyle, listed as Canada and the United States, shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.222: 2009: George E. Smith of the United States shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.223: 2010: Andre Geim, listed as Russia, the United Kingdom and the Netherlands, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.224: 2010: Konstantin Novoselov, listed as Russia and the United Kingdom, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.225: 2011: Saul Perlmutter of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.226: 2011: Brian P. Schmidt, listed as Australia and the United States, shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.227: 2011: Adam G. Riess of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.228: 2012: Serge Haroche of France shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.229: 2012: David J. Wineland of the United States shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.230: 2013: François Englert of Belgium shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.231: 2013: Peter Higgs of the United Kingdom shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.232: 2014: Isamu Akasaki of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.233: 2014: Hiroshi Amano of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.234: 2014: Shuji Nakamura, listed as Japan and the United States, shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.235: 2015: Takaaki Kajita of Japan shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.236: 2015: Arthur B. McDonald of Canada shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.237: 2016: David J. Thouless, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.238: 2016: F. Duncan M. Haldane, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.239: 2016: John M. Kosterlitz, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.240: 2017: Rainer Weiss, listed as Germany and the United States, shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.241: 2017: Kip Thorne of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.242: 2017: Barry Barish of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
Sentence 1: The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .
  Fact 1.0: The first Nobel Prize in Physics was awarded in 1901
  Fact 1.1: The first Nobel Prize in Physics went to Wilhelm Conrad Röntgen of Germany
  Fact 1.2: Wilhelm Conrad Röntgen received 150,782 SEK
  Fact 1.3: 150,782 SEK is equal to 7,731,004 SEK in December 2007
  Fact 1.4: John Bardeen is the only laureate to win the prize twice
  Fact 1.5: John Bardeen won the prize in 1956 and in 1972
  Fact 1.6: Maria Skłodowska-Curie also won two Nobel Prizes
  Fact 1.7: Maria Skłodowska-Curie won the physics prize in 1903
  Fact 1.8: Maria Skłodowska-Curie won the chemistry prize in 1911
  Fact 1.9: William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate
  Fact 1.10: William Lawrence Bragg won the prize in 1915 at the age of 25
  Fact 1.11: Two women have won the prize: Curie and Maria Goeppert-Mayer
  Fact 1.12: Maria Goeppert-Mayer won the prize in 1963
  Fact 1.13: As of 2017, the prize has been awarded to 206 individuals
  Fact 1.14: There have been six years in which the Nobel Prize in Physics was not awarded: 1916, 1931, 1934, 1940, 1941 and 1942

Retrieved items:
<item 1>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1925 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1926 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1927 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1928 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1929 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1930 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1932 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 1>

<item 2>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1910 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1911 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1912 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1913 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1914 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1915 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1917 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1918 '' .
</item 2>

<item 3>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1918 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1919 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1920 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1921 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1922 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1923 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1924 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1925 '' .
</item 3>

<item 4>
List of Nobel laureates in Physics
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1904 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1905 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1906 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1907 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1908 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1909 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1910 '' . Nobel Foundation .
</item 4>

<item 5>
List of Nobel laureates in Physics
List of Nobel laureates in Physics - wikipedia List of Nobel laureates in Physics Jump to : navigation , search Front side ( obverse ) of the Nobel Prize Medal
for Physics presented to Edward Victor Appleton in 1947 The Nobel Prize in Physics ( Swedish : Nobelpriset i fysik ) is awarded annually by the Royal Swedish
Academy of Sciences to scientists in the various fields of physics . It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel ( who died
in 1896 ) , awarded for outstanding contributions in physics . As dictated by Nobel 's will , the award is administered by the Nobel Foundation and awarded by a
committee that consists of five members elected by the Royal Swedish Academy of Sciences . The award is presented in Stockholm at an annual ceremony on December
10 , the anniversary of Nobel 's death . Each recipient receives a medal , a diploma and a monetary award prize that has varied throughout the years .
</item 5>

<item 6>
List of Nobel laureates in Physics
Contents ( hide ) 1 Statistics 2 Laureates 3 See also 4 References 4.1 Notes 4.2 Citations 4.3 Sources 5 External links Statistics ( edit ) The first Nobel
Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John
Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry
in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won
the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals .
</item 6>

<item 7>
List of Nobel laureates in Physics
This information may not necessarily reflect the recipient 's birthplace or citizenship . ^ C. The citation for each award is quoted ( not always in full ) from
nobelprize.org , the official website of the Nobel Foundation . The links in this column are to articles ( or sections of articles ) on the history and areas of
physics for which the awards were presented . The links are intended only as a guide and explanation . For a full account of the work done by each Nobel
laureate , please see the biography articles linked from the name column . Citations ( edit ) Jump up ^ `` Alfred Nobel -- The Man Behind the Nobel Prize '' .
Nobel Foundation . Retrieved 2008 - 10 - 29 . Jump up ^ `` The Nobel Prize Awarders '' . Nobel Foundation . Archived from the original on 2008 - 10 - 15 .
Retrieved 2008 - 10 - 29 . Jump up ^ `` The Nobel Prize Award Ceremonies '' . Nobel Foundation . Archived from the original on 2008 - 08 - 22 .
</item 7>

<item 8>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1943 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1944 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1945 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1946 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1947 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1948 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1949 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1950 '' .
</item 8>

<item 9>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 2016 '' . Nobel Foundation . Retrieved 2016 - 10 - 04 . Jump up ^ `` The Nobel Prize in Physics 2017 '' . Nobel
Foundation . Retrieved 2017 - 10 - 03 . Sources ( edit ) `` All Nobel Laureates in Physics '' . Nobel Foundation . Retrieved 2008 - 10 - 08 . External links (
edit ) Wikimedia Commons has media related to Nobel laureates in Physics . Official website of the Royal Swedish Academy of Sciences Official website of the
Nobel Foundation Laureates of the Nobel Prize in Physics 1901 -- 1925 1901 Röntgen 1902 Lorentz / Zeeman 1903 Becquerel / P. Curie / M. Curie 1904 Rayleigh 1905
Lenard 1906 J.J.
</item 9>

<item 10>
List of Nobel laureates in Physics
As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934
, 1940 -- 1942 ) . Laureates ( edit ) Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has
rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service
they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France ``
for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri
Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and
for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays ''
1906 Joseph
</item 10>
=== END ITEM id=b46555a209fe2d2f ===
