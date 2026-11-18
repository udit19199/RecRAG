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

=== ITEM id=901a71ff9ff328fe ===
Gold sentences and facts:
Sentence 0: The National Legal Aid & Defender Association ( NLADA ) is the oldest and largest national , nonprofit membership organization devoted to advocating equal justice for all Americans and was established in 1911 .
  Fact 0.0: The National Legal Aid & Defender Association (NLADA) is the oldest national membership organization.
  Fact 0.1: The National Legal Aid & Defender Association (NLADA) is the largest national membership organization.
  Fact 0.2: The National Legal Aid & Defender Association (NLADA) is nonprofit.
  Fact 0.3: The National Legal Aid & Defender Association (NLADA) is devoted to advocating equal justice for all Americans.
  Fact 0.4: The National Legal Aid & Defender Association (NLADA) was established in 1911.

Retrieved items:
<item 1>
American Bar Association Model Rules of Professional Conduct
However , the Model Rules have been adopted , in whole or in part , and sometimes in variation , as the rules of professional conduct for attorneys in 49 states
, the District of Columbia , and four of the five inhabited U.S. territories . The rules of professional conduct or professional responsibility adopted in a
particular state are legally enforceable against the lawyers of that state as well as any lawyer practicing there on a pro hac vice basis . On December 17 ,
2008 , New York announced that it would finally abandon the old ABA Model Code of Professional Responsibility ( the last state to do so ) and adopt a heavily
modified version of the ABA Model Rules , effective April 1 , 2009 . New York 's version of the Model Rules was created by adjusting the standard Model Rules to
reflect indigenous New York rules that had been incorporated over the years into its version of the Model Code .
</item 1>

<item 2>
American Bar Association Model Rules of Professional Conduct
In 2001 , the Commission for the Revision of the Rules of Professional Conduct of the State Bar of California began a comprehensive revision of the California
rules that would , among other things , convert them into a localized version of the ABA Model Rules . However , the Commission 's progress had been extremely
slow , simply because there are so many substantive and structural differences between the California rules and the ABA models . In October 2010 , the State Bar
of California formally adopted the Commission 's report , recommending adoption of 67 proposed changes to harmonise the California Rules of Professional Conduct
with the ABA Model Rules of Professional Responsibility . On September 19 , 2014 , however , for reasons that were not fully explained , the Supreme Court of
California rejected all of the proposed revisions and returned them to the State Bar . The Court 's letter directed the State Bar to start the process all over
again with a new Commission , and to submit a new set of revised rules by March 31 , 2017 . Professional regulation ( edit ) Except for the Patent Bar , there
is no national legal profession in the United States but there are 56 separate legal professions .
</item 2>

<item 3>
American Bar Association Model Rules of Professional Conduct
In contrast to most Commonwealth countries , where transactional lawyers and litigators are separately licensed as `` solicitors '' and `` barristers , ''
respectively ; all American attorneys are licensed to practice as both solicitors and barristers ( therefore these terms are almost never used ) . Likewise ,
whereas solicitors in England and Wales are professionally self - regulated by the private - sector Law Society of England and Wales while barristers are self -
regulated by the private - sector General Council of the Bar , attorneys in all states except California , all territories and the District of Columbia are
generally regulated by the public sector judicial branch of state , district or territorial government , as officers of the court . The legal profession in
California is regulated jointly by the California Supreme Court , mostly through its agency the State Bar of California and by the California Legislature ,
which exercises legislative control over all professions . Within the public sector regulatory agencies , boards and organisations , officers are invariably
licensed members of the pertinent profession while key managerial and general staff typically are not . The Executive Director and Deputy Executive Director of
the State Bar of California , for example , are active members of the State Bar .
</item 3>

<item 4>
American Bar Association Model Rules of Professional Conduct
This is due to California law and public policy being so markedly different from the rest of the country in so many ways . For example , California is one of
only four states where covenants not to compete are void as against public policy ; and it is one of only three states where trusts are presumed to be revocable
unless the trust instrument explicitly states the trust is irrevocable ( whereas the opposite is true in the other 47 states ) . By way of comparative example ,
California has long held the position that clients must be encouraged to speak candidly to their legal counsel about both past and future actions knowing their
statements will be held in complete confidence . Their counsel can then explain the relevant law to them and urge them to conform their future conduct to the
law , thus promoting lawful conduct . In contrast , ABA Model Rule 1.6 contains exceptions to confidentiality for both violent and nonviolent ( e.g. financial )
future crimes , which California believes creates a strong disincentive for clients to discuss their future plans with their attorneys - but for which the
client might chose to comply with the law instead of violating it .
</item 4>

<item 5>
American Bar Association Model Rules of Professional Conduct
California ( edit ) California still has its own set of unique rules governing professional attorney conduct . Although many of the rules are similar and
sometimes identical to the ABA Model Rules , the California rules differ markedly from the ABA models in both format , scope and content . For example , under
the ABA Model Rules , an attorney may be disciplined for the appearance of impropriety , regardless whether any actual impropriety occurred ; whereas the
California rules permit discipline for only actual impropriety . This reflects the nature of California law . It is unlikely that California will ever adopt the
ABA Model Rules apart beyond harmonising the format and existing California rules that are identical or nearly identical to ABA Model Rules . Reasons for this
reluctance include the fact there are more than 280,000 attorneys licensed in California , so a drastic change from the existing rules to completely different
rules would shock the profession and , ultimately , would most likely fail . Additionally , many of California 's rules of professional conduct are established
by statute and , thus , would require action by the California Legislature to change them ; or , alternatively , action by the people of California through the
voter initiative process .
</item 5>

<item 6>
American Bar Association Model Rules of Professional Conduct
As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern
Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside
of confederation has not adopted them but instead has its own Código de Ética Profesional . Contents ( hide ) 1 State , commonwealth and territorial adoptions
of the Rules 1.1 California 2 Professional regulation 3 See also 4 References 5 External links State , Commonwealth and territorial adoptions of the Rules (
edit ) The American Bar Association is a private sector voluntary professional association . As such , it has no lawmaking power or regulatory authority , but
it is just like any other nongovernmental bar association . Accordingly , the Model Rules are not legally binding on ABA members or within any particular
jurisdiction .
</item 6>

<item 7>
American Bar Association Model Rules of Professional Conduct
American Bar Association Model Rules of professional conduct - wikipedia American Bar Association Model Rules of professional conduct Jump to : navigation ,
search Professional responsibility Duties to the client Confidentiality Avoiding conflict of interest Due diligence and Competence ( law ) Avoid commingling
Avoid self - dealing Effective assistance Avoid fee splitting Withdrawal from representation Duties to the court Disclosure of perjury Disclosure of adverse
authority Duties to the profession Limitations on legal advertising Report misconduct Sources of law ABA Model Rules Penalties for misconduct Disbarment
Judicial misconduct The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline
standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the
recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves
binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as
well as the day - to - day application of such rules .
</item 7>

<item 8>
American Bar Association Model Rules of Professional Conduct
While either of their secretaries could be a member of the State Bar , they most likely are not . Similarly , the Board of Trustees that oversee the business
operations of the State Bar of California under supervision of the California Supreme Court is legally required to be a mix of attorneys licensed to practice in
California and `` public members '' who may not be licensed attorneys . So , while American attorneys are regulated by public sector agencies instead of private
sector law societies , most of the people who develop the regulatory policies are practising attorneys and sitting judges , so there is a degree of self -
regulation but not nearly to the same degree that solicitors and barristers in England and Wales , and other Commonwealth jurisdictions are self - regulated by
private sector associations . See also ( edit ) American Bar Association Model Code of Professional Responsibility American Bar Association References ( edit )
Jump up ^ Press Release : New Attorney Rules of Professional Conduct Announced , Communications Office of the New York Courts , 17 December 2008 .
</item 8>

<item 9>
American Bar Association Model Rules of Professional Conduct
Another important factor is that the legal profession is regulated very differently in California than it is in the rest of the United States . Outside of
California , the profession is regulated solely by the judicial branch of state government . Within California , the State Bar of California operates as a state
government agency under the supervision of the California Supreme Court ; but the State Bar was created by the California Legislature , and the Legislature has
full power to regulate the legal profession , if it so wishes . In particular , the State Bar can not collect annual member dues unless and until the California
Legislature passes an annual State Bar Dues Bill , which the governor must sign into law . When Governor Pete Wilson vetoed the dues bill in 1997 , the State
Bar effectively shut down operations except for self - funded activities like the California Bar Exam . In 2010 , Governor Arnold Schwarzenegger vetoed the dues
bill but a crisis was averted on that occasion . But the most important reason the California Bar is reluctant to adopt the ABA Model Rules is that most members
believe the ABA Model Rules would not sufficiently protect the California public from harm .
</item 9>

<item 10>
American Bar Association Model Rules of Professional Conduct
External links ( edit ) The ABA Model Rules of Professional Conduct Pre-2002 Changes to the Model Rules since its adoption in 1983 from Cornell Law School
Retrieved from `` https://en.wikipedia.org/w/index.php?title=American_Bar_Association_Model_Rules_of_Professional_Conduct&oldid=789593953 '' Categories : 1983
in law Legal ethics Codes of conduct American Bar Association United States law 1983 documents Hidden categories : Articles containing potentially dated
statements from 2015 All articles containing potentially dated statements Talk Contents About Wikipedia 한국어 Suomi Edit links This page was last edited on 8 July
2017 , at 08 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=901a71ff9ff328fe ===

=== ITEM id=b679aef3b68d85de ===
Gold sentences and facts:
Sentence 0: Dual - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) Age of Child Housing Food Transport . Clothing Health Child Care / Education Misc . Total Before - tax income : Less than $59,410 ( Average = $38,000 ) 0 to 2 2,990 1,160 1,170 640 630 2,040 420 9,050 3 to 5 2,990 1,260 1,230 500 590 1,910 620 9,100 6 to 8 2,990 1,710 1,350 570 660 1,290 630 8,760 9 to 11 2,990 1,970 1,350 580 710 1,910 630 9,520 12 to 14 2,990 2,130 1,480 690 1,090 1,110 700 9,960 15 to 17 2,990 2,120 1,630 730 1,010 1,290 580 9,970 Total 53,820 31,050 24,630 11,130 14,070 23,640 10,740 169,080 Before - tax income : $59,410 to $102,870 ( Average = $79,940 ) 0 to 2 3,920 1,405 1,690 760 850 2,860 890 12,370 3 to 5 3,920 1,490 1,740 610 800 2,740 1,090 12,390 6 to 8 3,920 2,100 1,860 680 940 1,680 1,110 12,290 9 to 11 3,920 2,400 1,870 710 1,000 2,110 1,100 13,110 12 to 14 3,920 2,580 1,990 840 1,410 1,910 1,170 13,820 15 to 17 3,920 2,570 2,150 900 1,330 2,400 1,050 14,320 Total 70,560 37,620 33,900 13,500 18,990 41,100 19,230 234,900 Before - tax income : More than $102,870 ( Average = $180,040 ) 0 to 2 7,100 1,900 2,550 1,050 980 5,090 1,790 20,460 3 to 5 7,100 2,000 2,610 880 930 4,970 1,990 20,480 6 to 8 7,100 2,630 2,730 970 1,080 3,910 2,000 20,420 9 to 11 7,100 2,980 2,730 1,010 1,150 4,350 2,000 21,320 12 to 14 7,100 3,190 2,860 1,170 1,610 4,700 2,070 22,700 15 to 17 7,100 3,180 3,020 1,280 1,520 6,460 1,950 24,510 Total 127,800 47,640 49,500 19,080 21,810 88,440 35,400 389,670
  Fact 0.0: The table gives USDA average spending per child for dual-parent families, not the basic cost of raising a child
  Fact 0.1: Spending categories are housing, food, transport, clothing, health, child care and education, miscellaneous, and total
  Fact 0.2: Dual-parent families with before-tax income under $59,410 have an average before-tax income of $38,000
  Fact 0.3: Dual-parent families with before-tax income under $59,410 and a child aged 0 to 2 spend $2,990 on housing, $1,160 on food, $1,170 on transport, $640 on clothing, $630 on health, $2,040 on child care and education, $420 on miscellaneous, totaling $9,050
  Fact 0.4: Dual-parent families with before-tax income under $59,410 and a child aged 3 to 5 spend $2,990 on housing, $1,260 on food, $1,230 on transport, $500 on clothing, $590 on health, $1,910 on child care and education, $620 on miscellaneous, totaling $9,100
  Fact 0.5: Dual-parent families with before-tax income under $59,410 and a child aged 6 to 8 spend $2,990 on housing, $1,710 on food, $1,350 on transport, $570 on clothing, $660 on health, $1,290 on child care and education, $630 on miscellaneous, totaling $8,760
  Fact 0.6: Dual-parent families with before-tax income under $59,410 and a child aged 9 to 11 spend $2,990 on housing, $1,970 on food, $1,350 on transport, $580 on clothing, $710 on health, $1,910 on child care and education, $630 on miscellaneous, totaling $9,520
  Fact 0.7: Dual-parent families with before-tax income under $59,410 and a child aged 12 to 14 spend $2,990 on housing, $2,130 on food, $1,480 on transport, $690 on clothing, $1,090 on health, $1,110 on child care and education, $700 on miscellaneous, totaling $9,960
  Fact 0.8: Dual-parent families with before-tax income under $59,410 and a child aged 15 to 17 spend $2,990 on housing, $2,120 on food, $1,630 on transport, $730 on clothing, $1,010 on health, $1,290 on child care and education, $580 on miscellaneous, totaling $9,970
  Fact 0.9: Dual-parent families with before-tax income under $59,410 have totals across all age groups of $53,820 housing, $31,050 food, $24,630 transport, $11,130 clothing, $14,070 health, $23,640 child care and education, $10,740 miscellaneous, and $169,080 overall
  Fact 0.10: Dual-parent families with before-tax income of $59,410 to $102,870 have an average before-tax income of $79,940
  Fact 0.11: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 0 to 2 spend $3,920 on housing, $1,405 on food, $1,690 on transport, $760 on clothing, $850 on health, $2,860 on child care and education, $890 on miscellaneous, totaling $12,370
  Fact 0.12: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 3 to 5 spend $3,920 on housing, $1,490 on food, $1,740 on transport, $610 on clothing, $800 on health, $2,740 on child care and education, $1,090 on miscellaneous, totaling $12,390
  Fact 0.13: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 6 to 8 spend $3,920 on housing, $2,100 on food, $1,860 on transport, $680 on clothing, $940 on health, $1,680 on child care and education, $1,110 on miscellaneous, totaling $12,290
  Fact 0.14: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 9 to 11 spend $3,920 on housing, $2,400 on food, $1,870 on transport, $710 on clothing, $1,000 on health, $2,110 on child care and education, $1,100 on miscellaneous, totaling $13,110
  Fact 0.15: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 12 to 14 spend $3,920 on housing, $2,580 on food, $1,990 on transport, $840 on clothing, $1,410 on health, $1,910 on child care and education, $1,170 on miscellaneous, totaling $13,820
  Fact 0.16: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 15 to 17 spend $3,920 on housing, $2,570 on food, $2,150 on transport, $900 on clothing, $1,330 on health, $2,400 on child care and education, $1,050 on miscellaneous, totaling $14,320
  Fact 0.17: Dual-parent families with before-tax income of $59,410 to $102,870 have totals across all age groups of $70,560 housing, $37,620 food, $33,900 transport, $13,500 clothing, $18,990 health, $41,100 child care and education, $19,230 miscellaneous, and $234,900 overall
  Fact 0.18: Dual-parent families with before-tax income over $102,870 have an average before-tax income of $180,040
  Fact 0.19: Dual-parent families with before-tax income over $102,870 and a child aged 0 to 2 spend $7,100 on housing, $1,900 on food, $2,550 on transport, $1,050 on clothing, $980 on health, $5,090 on child care and education, $1,790 on miscellaneous, totaling $20,460
  Fact 0.20: Dual-parent families with before-tax income over $102,870 and a child aged 3 to 5 spend $7,100 on housing, $2,000 on food, $2,610 on transport, $880 on clothing, $930 on health, $4,970 on child care and education, $1,990 on miscellaneous, totaling $20,480
  Fact 0.21: Dual-parent families with before-tax income over $102,870 and a child aged 6 to 8 spend $7,100 on housing, $2,630 on food, $2,730 on transport, $970 on clothing, $1,080 on health, $3,910 on child care and education, $2,000 on miscellaneous, totaling $20,420
  Fact 0.22: Dual-parent families with before-tax income over $102,870 and a child aged 9 to 11 spend $7,100 on housing, $2,980 on food, $2,730 on transport, $1,010 on clothing, $1,150 on health, $4,350 on child care and education, $2,000 on miscellaneous, totaling $21,320
  Fact 0.23: Dual-parent families with before-tax income over $102,870 and a child aged 12 to 14 spend $7,100 on housing, $3,190 on food, $2,860 on transport, $1,170 on clothing, $1,610 on health, $4,700 on child care and education, $2,070 on miscellaneous, totaling $22,700
  Fact 0.24: Dual-parent families with before-tax income over $102,870 and a child aged 15 to 17 spend $7,100 on housing, $3,180 on food, $3,020 on transport, $1,280 on clothing, $1,520 on health, $6,460 on child care and education, $1,950 on miscellaneous, totaling $24,510
  Fact 0.25: Dual-parent families with before-tax income over $102,870 have totals across all age groups of $127,800 housing, $47,640 food, $49,500 transport, $19,080 clothing, $21,810 health, $88,440 child care and education, $35,400 miscellaneous, and $389,670 overall

Retrieved items:
<item 1>
Cost of raising a child
880 3,080 2,370 650 1,560 4,150 1,980 19,670 15 to 17 5,880 3,220 2,370 730 1,550 5,010 1,810 20,570 Total 105,840 48,390 40,140 10,830 22,410 72,810 33,000
333,420 All numbers are in US dollars . India ( edit ) Based on an estimate by Economic Times in April 2011 , the cost of raising a child from birth to age of
majority ( 21 Years ) for a middle to upper - middle income family comes to about ₹ 55 lakh ( US $87,000 ) in total . Cost break up is as follows : Cost of
raising a child in India ( as of April , 2011 ) Expenditure Class Estimated Cost ( Rs .
</item 1>

<item 2>
Cost of raising a child
Nor does it offer any spending estimates if the child remains in the home as a dependent after the age of 18 . Both tables are for the United States overall ,
not based on any specific region in the country . Dual - parent family - USDA Average spending per child ( not the basic cost of raising ) ( edit ) Dual -
Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) Age of Child Housing Food Transport . Clothing Health Child Care / Education
Misc . Total Before - tax income : Less than $59,410 ( Average = $38,000 ) 0 to 2 2,990 1,160 1,170 640 630 2,040 420 9,050 3 to 5 2,990 1,260 1,230 500 590
1,910 620 9,100 6 to 8 2,990 1,710 1,350 570 660 1,290 630 8,760 9 to 11 2,990 1,970 1,350 580 710 1,910 630 9,
</item 2>

<item 3>
Cost of raising a child
www.lv.com . Retrieved 2016 - 06 - 27 . Jump up ^ Friday , H. Swint ( July 20 , 2007 ) . `` Cost of raising children not as high as government would have you
believe '' . caller.com . Retrieved March 31 , 2013 . Jump up ^ http://abcnews.go.com/Business/page?id=4019746 ^ Jump up to : Lino , Mark . `` Expenditures on
Children by Families , 2011 ( Miscellaneous Publication Number 1528 - 2011 ) '' ( PDF ) . United States Department of Agriculture , Center for Nutrition Policy
and Promotion . Retrieved August 5 , 2012 . Jump up ^ `` How much it cost to raise a child '' ( pdf ) . report . Economic Times . 18 April 2011 . Retrieved 13
July 2016 . External links ( edit ) `` Raising that ' 07 baby will cost $204,060 in U.S. '' - from Reuters , retrieved August 5 , 2012 .
</item 3>

<item 4>
Cost of raising a child
480 6 to 8 7,100 2,630 2,730 970 1,080 3,910 2,000 20,420 9 to 11 7,100 2,980 2,730 1,010 1,150 4,350 2,000 21,320 12 to 14 7,100 3,190 2,860 1,170 1,610 4,700
2,070 22,700 15 to 17 7,100 3,180 3,020 1,280 1,520 6,460 1,950 24,510 Total 127,800 47,640 49,500 19,080 21,810 88,440 35,400 389,670 All numbers are in US
dollars . Also notice that the dual parent and solo parent charts use different income ranges so that direct comparison between the two should be undertaken
with care .
</item 4>

<item 5>
Cost of raising a child
Cost of raising a child - wikipedia Cost of raising a child Jump to : navigation , search The cost of raising a child varies from country to country . The cost
of raising a child is usually determined according to a formula that accounts for major areas of expenditure , such as food , housing , and clothing . However ,
any given family 's actual expenses may differ from the estimates . For example , the rent on a home does not usually change when the tenants have another child
, so the family 's housing costs may remain the same . In other cases , the home may be too small , in which case the family might move to a larger home at a
higher cost . Contents ( hide ) 1 Developing countries 2 United Kingdom 3 United States 3.1 Dual - Parent Family - USDA Average Spending per Child ( not the
basic cost of raising ) 3.2 Single - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) 4 India 5 See also 6 References 7
External links Developing countries ( edit ) According to Globalissues.org , `` Almost half the world -- over three billion people -- live on less than $2.50 a
day . ''
</item 5>

<item 6>
Cost of raising a child
Results have been weighted to nationally representative criteria United States ( edit ) The tax exemption per child is $3,900 in 2013 which is based on the
basic cost of raising a child . Plus there is a tax credit up to $1000 for each child. citation needed Based on a survey by the U.S. Department of Agriculture ,
the table below shows the estimated Average Spending on Children by Families . The data comes from the Consumer Expenditure Survey by the U.S. Department of
Labor , conducted from 2005 - 06 . The figures have been updated to 2011 dollars using the Consumer Price Index . However , some dispute the numbers as being
biased high for political reasons ( e.g. , Texas A&M University Finance Professor H. Swint Friday : `` The numbers , reported by the U.S. Department of
Agriculture , are outrageously misleading . Often government statistics are produced for political objectives that cause the research methodology to be biased
toward finding the highest dollar amount to support the objective . '' ) . These figures from the USDA go up to age 18 , and do not include any college or
university education .
</item 6>

<item 7>
Cost of raising a child
440 15 to 17 2,840 2,270 1,130 460 930 880 670 9,180 Total 51,120 33,090 17,850 7,080 12,840 22,980 12,450 157,410 Before - tax income : $59,410 or more (
Average = $107,820 ) 0 to 2 5,880 2,080 1,920 590 980 3,670 1,650 16,770 3 to 5 5,880 2,070 2,160 500 1,090 4,210 1,750 17,660 6 to 8 5,880 2,680 2,260 530
1,180 3,350 1,930 17,810 9 to 11 5,880 3,000 2,300 610 1,110 3,880 1,880 18,660 12 to 14 5,880 3,080 2,370 650 1,560 4,
</item 7>

<item 8>
Cost of raising a child
Single - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) ( edit ) Single - Parent Family - USDA Average Spending per Child (
not the basic cost of raising ) Age of Child Housing Food Transport . Clothing Health Child care / Education Misc . Total Before - tax income : Less than
$40,410 ( Average = $18,350 ) 0 to 2 2,840 1,400 680 410 520 1,400 510 7,760 3 to 5 2,840 1,370 920 330 600 1,940 610 8,610 6 to 8 2,840 1,830 1,030 340 670
1,940 780 8,450 9 to 11 2,840 2,010 1,060 400 620 1,360 740 9,030 12 to 14 2,840 2,150 1,130 420 940 1,120 840 9,440 15 to 17 2,840 2,270 1,
</item 8>

<item 9>
Cost of raising a child
110 12,290 9 to 11 3,920 2,400 1,870 710 1,000 2,110 1,100 13,110 12 to 14 3,920 2,580 1,990 840 1,410 1,910 1,170 13,820 15 to 17 3,920 2,570 2,150 900 1,330
2,400 1,050 14,320 Total 70,560 37,620 33,900 13,500 18,990 41,100 19,230 234,900 Before - tax income : More than $102,870 ( Average = $180,040 ) 0 to 2 7,100
1,900 2,550 1,050 980 5,090 1,790 20,460 3 to 5 7,100 2,000 2,610 880 930 4,970 1,990 20,480 6 to 8 7,100 2,630 2,
</item 9>

<item 10>
Cost of raising a child
970 1,350 580 710 1,910 630 9,520 12 to 14 2,990 2,130 1,480 690 1,090 1,110 700 9,960 15 to 17 2,990 2,120 1,630 730 1,010 1,290 580 9,970 Total 53,820 31,050
24,630 11,130 14,070 23,640 10,740 169,080 Before - tax income : $59,410 to $102,870 ( Average = $79,940 ) 0 to 2 3,920 1,405 1,690 760 850 2,860 890 12,370 3
to 5 3,920 1,490 1,740 610 800 2,740 1,090 12,390 6 to 8 3,920 2,100 1,860 680 940 1,680 1,110 12,290 9 to 11 3,920 2,
</item 10>
=== END ITEM id=b679aef3b68d85de ===

=== ITEM id=fb7fadca2dfc7168 ===
Gold sentences and facts:
Sentence 0: On the Origin of Species ( or more completely , On the Origin of Species by Means of Natural Selection , or the Preservation of Favoured Races in the Struggle for Life ) , published on 24 November 1859 , is a work of scientific literature by Charles Darwin which is considered to be the foundation of evolutionary biology . Darwin 's book introduced the scientific theory that populations evolve over the course of generations through a process of natural selection . It presented a body of evidence that the diversity of life arose by common descent through a branching pattern of evolution . Darwin included evidence that he had gathered on the Beagle expedition in the 1830s and his subsequent findings from research , correspondence , and experimentation .
  Fact 0.0: The full title of On the Origin of Species is On the Origin of Species by Means of Natural Selection, or the Preservation of Favoured Races in the Struggle for Life.
  Fact 0.1: On the Origin of Species was published on 24 November 1859.
  Fact 0.2: On the Origin of Species is a work of scientific literature by Charles Darwin.
  Fact 0.3: On the Origin of Species is considered to be the foundation of evolutionary biology.
  Fact 0.4: Charles Darwin's book On the Origin of Species introduced the scientific theory that populations evolve over the course of generations.
  Fact 0.5: The scientific theory in On the Origin of Species holds that populations evolve through a process of natural selection.
  Fact 0.6: On the Origin of Species presented a body of evidence.
  Fact 0.7: The evidence in On the Origin of Species holds that the diversity of life arose by common descent.
  Fact 0.8: The evidence in On the Origin of Species holds that this common descent follows a branching pattern of evolution.
  Fact 0.9: Charles Darwin included evidence he had gathered on the Beagle expedition in the 1830s in On the Origin of Species.
  Fact 0.10: Charles Darwin included his subsequent findings from research in On the Origin of Species.
  Fact 0.11: Charles Darwin included his subsequent findings from correspondence in On the Origin of Species.
  Fact 0.12: Charles Darwin included his subsequent findings from experimentation in On the Origin of Species.
Sentence 1: On the Origin of Species The title page of the 1859 edition of On the Origin of Species Author Charles Darwin Country United Kingdom of Great Britain and Ireland Language English Subject Natural selection Evolutionary biology Published 24 November 1859 ( John Murray ) Media type Print ( Hardback ) Pages 502 OCLC 352242 Preceded by On the Tendency of Species to form Varieties ; and on the Perpetuation of Varieties and Species by Natural Means of Selection Followed by Fertilisation of Orchids
  Fact 1.0: On the Origin of Species has the author Charles Darwin.
  Fact 1.1: On the Origin of Species has the country United Kingdom of Great Britain and Ireland.
  Fact 1.2: On the Origin of Species is written in English.
  Fact 1.3: On the Origin of Species has the subject natural selection.
  Fact 1.4: On the Origin of Species has the subject evolutionary biology.
  Fact 1.5: On the Origin of Species was published on 24 November 1859 by John Murray.
  Fact 1.6: The title page of the 1859 edition of On the Origin of Species is shown.
  Fact 1.7: The media type of On the Origin of Species is print (hardback).
  Fact 1.8: On the Origin of Species has 502 pages.
  Fact 1.9: On the Origin of Species has OCLC number 352242.
  Fact 1.10: On the Origin of Species was preceded by On the Tendency of Species to form Varieties; and on the Perpetuation of Varieties and Species by Natural Means of Selection.
  Fact 1.11: On the Origin of Species was followed by Fertilisation of Orchids.

Retrieved items:
<item 1>
On the Origin of Species
There was much less controversy than had greeted the 1844 publication Vestiges of Creation , which had been rejected by scientists , but had influenced a wide
public readership into believing that nature and human society were governed by natural laws . The Origin of Species as a book of wide general interest became
associated with ideas of social reform . Its proponents made full use of a surge in the publication of review journals , and it was given more popular attention
than almost any other scientific work , though it failed to match the continuing sales of Vestiges . Darwin 's book legitimised scientific discussion of
evolutionary mechanisms , and the newly coined term Darwinism was used to cover the whole range of evolutionism , not just his own ideas . By the mid-1870s ,
evolutionism was triumphant . While Darwin had been somewhat coy about human origins , not identifying any explicit conclusion on the matter in his book , he
had dropped enough hints about human 's animal ancestry for the inference to be made , and the first review claimed it made a creed of the `` men from monkeys
'' idea from Vestiges .
</item 1>

<item 2>
On the Origin of Species
concepts Germ theory of disease Central dogma of molecular biology Darwinism Great chain of being Hierarchy of life Lamarckism One gene -- one enzyme hypothesis
Protocell RNA world hypothesis Sequence hypothesis Spontaneous generation History Classical antiquity Aristotle Aristotle 's biology On Generation and
Corruption History of Animals Theophrastus Historia Plantarum Dioscorides De Materia Medica Galen Renaissance , Early Modern Conrad Gessner Historia animalium
Andreas Vesalius De humani corporis fabrica William Harvey De Motu Cordis Frederik Ruysch Jan Swammerdam Antonie van Leeuwenhoek Robert Hooke Micrographia
Francesco Redi Evolution 19th century Linnaeus Systema Naturae Buffon Histoire Naturelle Lamarck Philosophie Zoologique Humboldt Charles Lyell Principles of
Geology Charles Darwin On the Origin of Species The Descent of Man Gregor Mendel Alfred Russel Wallace Martinus Beijerinck Henry Walter Bates Modern synthesis
William Bateson Theodosius Dobzhansky Genetics and the Origin of Species R.A. Fisher E.B. Ford J.B.S.
</item 2>

<item 3>
On the Origin of Species
In a survey conducted by a group of academic booksellers , publishers and librarians in advance of Academic Book Week in the United Kingdom , On the Origin of
Species was voted the most influential academic book ever written . It was hailed as `` the supreme demonstration of why academic books matter '' and `` a book
which has changed the way we think about everything '' . See also ( edit ) Book : Evolution On the Origin of Species -- full text at Wikisource of the first
edition , 1859 The Origin of Species -- full text at Wikisource of the 6th edition , 1872 Charles Darwin bibliography History of biology History of evolutionary
thought History of speciation Modern evolutionary synthesis The Complete Works of Charles Darwin Online The Descent of Man , and Selection in Relation to Sex ,
published in 1871 ; his second major book on evolutionary theory . Transmutation of species Notes ( edit ) Jump up ^ Darwin 1859 , p. iii ^ Jump up to : Freeman
1977 ^ Jump up to : The book 's full original title was On the Origin of Species by Means of Natural Selection , or the Preservation of Favoured Races in the
Struggle for Life .
</item 3>

<item 4>
On the Origin of Species
It would have been useless and injurious to the success of the book to have paraded without giving any evidence my conviction with respect to his origin . ''
Jump up ^ Darwin 1871 , pp. 4 -- 5 , Quote : `` During many years it has seemed to me highly probable that sexual selection has played an important part in
differentiating the races of man ; but in my ' Origin of Species ' ( first edition , p. 199 ) I contented myself by merely alluding to this belief . '' Jump up
^ Browne 2002 , pp. 376 -- 379 ^ Jump up to : van Wyhe 2008 , pp. 48 -- 49 ^ Jump up to : Bowler 2003 , pp. 177 -- 180 Jump up ^ Darwin in letters , 1860 :
Answering critics Jump up ^ Browne 2002 , pp. 102 -- 103 Jump up ^ Darwin & Costa 2009 , p. 488 ^ Jump up to : Radick 2013 , pp. 174 -- 175 Huxley & Kettlewell
1965 , p.
</item 4>

<item 5>
On the Origin of Species
Barker , Lincolniana . Darwin 's notes on the creation of On the Origin of Species digitised in Cambridge Digital Library Charles Darwin Life Darwin -- Wedgwood
family Education Voyage on HMS Beagle Inception of theory Development of theory Publication of theory Reactions to On the Origin of Species Orchids to Variation
Descent of Man to Emotions Insectivorous Plants to Worms Portraits of Darwin Religious views Health Women Writings Extracts from Letters to Henslow The Voyage
of the Beagle Zoology of the Voyage of H.M.S. Beagle The Structure and Distribution of Coral Reefs On the Tendency of Species to form Varieties ; and on the
Perpetuation of Varieties and Species by Natural Means of Selection On the Origin of Species Fertilisation of Orchids Geological Observations on South America
Geological Observations on the Volcanic Islands The Variation of Animals and Plants under Domestication The Descent of Man , and Selection in Relation to Sex
The Expression of the Emotions in Man and Animals Insectivorous Plants The Power of Movement in Plants The Formation of Vegetable Mould through the Action of
Worms Autobiography Correspondence The Life and Letters of Charles Darwin More Letters of Charles Darwin
</item 5>

<item 6>
On the Origin of Species
On the Origin of species - wikipedia On the Origin of species Jump to : navigation , search `` Origin of Species '' redirects here . For other uses , see Origin
of Species ( disambiguation ) . On the Origin of Species The title page of the 1859 edition of On the Origin of Species Author Charles Darwin Country United
Kingdom of Great Britain and Ireland Language English Subject Natural selection Evolutionary biology Published 24 November 1859 ( John Murray ) Media type Print
( Hardback ) Pages 502 OCLC 352242 Preceded by On the Tendency of Species to form Varieties ; and on the Perpetuation of Varieties and Species by Natural Means
of Selection Followed by Fertilisation of Orchids Part of a series on Evolutionary biology Key topics ( show ) Introduction to evolution Common descent Evidence
of common descent Processes and outcomes ( show ) Population genetics Variation Diversity Mutation Natural selection Adaptation Polymorphism Genetic drift Gene
flow Speciation Adaptive radiation Co-operation Coevolution Divergence Convergence Parallel evolution Extinction Natural history ( show ) Origin of life History
of life Timeline of evolution Human evolution Phylogeny Biodiversity Biogeography Classification Evolutionary taxonomy Cladistics Transitional fossil Extinction
event History
</item 6>

<item 7>
On the Origin of Species
By 1977 , it had appeared in an additional 18 languages . Content ( edit ) Title pages and introduction ( edit ) John Gould 's illustration of Darwin 's rhea
was published in 1841 . The existence of two rhea species with overlapping ranges influenced Darwin . Page ii contains quotations by William Whewell and Francis
Bacon on the theology of natural laws , harmonising science and religion in accordance with Isaac Newton 's belief in a rational God who established a law -
abiding cosmos . In the second edition , Darwin added an epigraph from Joseph Butler affirming that God could work through scientific laws as much as through
miracles , in a nod to the religious concerns of his oldest friends . The Introduction establishes Darwin 's credentials as a naturalist and author , then
refers to John Herschel 's letter suggesting that the origin of species `` would be found to be a natural in contradistinction to a miraculous process '' : WHEN
on board HMS Beagle , as naturalist , I was much struck with certain facts in the distribution of the inhabitants of South America , and in the geological
relations of the present to the past inhabitants of that continent .
</item 7>

<item 8>
On the Origin of Species
Published anonymously . Murray , Andrew ( 1860 ) , `` On Mr Darwin 's Theory of the Origin of Species '' , Proceedings of the Royal Society of Edinburgh , 4 :
274 -- 291 . Owen , Richard ( 1860 ) , `` Review of Darwin 's Origin of Species '' , Edinburgh Review , 3 ( April 1860 ) : 487 -- 532 . Published anonymously .
Wilberforce , Samuel ( 1860 ) , `` ( Review of ) On the Origin of Species , by means of Natural Selection ; or the Preservation of Favoured Races in the
Struggle for Life '' , Quarterly Review , 108 ( 215 : July 1860 ) : 225 -- 264 . Published anonymously . For further reviews , see Darwin Online : Reviews &
Responses to Darwin , Darwin Online , 10 March 2009 , retrieved 18 June 2009 External links ( edit ) Wikimedia Commons has media related to The Origin of
Species .
</item 8>

<item 9>
On the Origin of Species
The second edition of 3,000 copies was quickly brought out on 7 January 1860 , and incorporated numerous corrections as well as a response to religious
objections by the addition of a new epigraph on page ii , a quotation from Charles Kingsley , and the phrase `` by the Creator '' added to the closing sentence
. During Darwin 's lifetime the book went through six editions , with cumulative changes and revisions to deal with counter-arguments raised . The third edition
came out in 1861 , with a number of sentences rewritten or added and an introductory appendix , An Historical Sketch of the Recent Progress of Opinion on the
Origin of Species , while the fourth in 1866 had further revisions . The fifth edition , published on 10 February 1869 , incorporated more changes and for the
first time included the phrase `` survival of the fittest '' , which had been coined by the philosopher Herbert Spencer in his Principles of Biology ( 1864 ) .
In January 1871 , George Jackson Mivart 's On the Genesis of Species listed detailed arguments against natural selection , and claimed it included false
metaphysics .
</item 9>

<item 10>
On the Origin of Species
He used the term broadly , and as well as discussions of `` the several races , for instance , of the cabbage '' and `` the hereditary varieties or races of our
domestic animals and plants '' , there are three instances in the book where the phrase `` races of man '' is used , referring to races of humans . Publication
and subsequent editions ( edit ) On the Origin of Species was first published on Thursday 24 November 1859 , priced at fifteen shillings with a first printing
of 1250 copies . The book had been offered to booksellers at Murray 's autumn sale on Tuesday 22 November , and all available copies had been taken up
immediately . In total , 1,250 copies were printed but after deducting presentation and review copies , and five for Stationers ' Hall copyright , around 1,170
copies were available for sale . Significantly , 500 were taken by Mudie 's Library , ensuring that the book promptly reached a large number of subscribers to
the library .
</item 10>
=== END ITEM id=fb7fadca2dfc7168 ===

=== ITEM id=30c5f2ca6e467d69 ===
Gold sentences and facts:
Sentence 0: A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called `` perfecting the security interest '' in the property , and this type of loan is a secured loan . A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit documents and the lease , respectively , and not the financing statement .
  Fact 0.0: A UCC-1 financing statement is a legal form.
  Fact 0.1: UCC-1 is an abbreviation for Uniform Commercial Code-1.
  Fact 0.2: A creditor files a UCC-1 financing statement to give notice that it has or may have an interest in the personal property of a debtor.
  Fact 0.3: A debtor is a person who owes a debt to the creditor, as typically specified in the agreement creating the debt.
  Fact 0.4: A financing statement is filed to perfect a creditor's security interest.
  Fact 0.5: The filing gives public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt, with a certain priority.
  Fact 0.6: Such notices of sale are often found in local newspapers.
  Fact 0.7: Once the form has been filed, the creditor establishes a relative priority with other creditors of the debtor.
  Fact 0.8: Perfecting the security interest is also called perfecting the security interest in the property.
  Fact 0.9: This type of loan is a secured loan.
  Fact 0.10: A financing statement may also be filed in the real estate records by a lessor of fixtures.
  Fact 0.11: The real estate filing establishes the lessor's priority against a holder of a mortgage or other lien on the real property.
  Fact 0.12: The creditor's rights against the debtor are based on the credit documents, not the financing statement.
  Fact 0.13: The lessor's rights against the lessee are based on the lease, not the financing statement.
Sentence 1: The financing statement is generally filed with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which is tasked with overseeing business organizations and activities , including receipt of financing statements . However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures . In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where third parties are most likely to search for such record .
  Fact 1.0: The financing statement is generally filed with the office of the state secretary of state.
  Fact 1.1: The financing statement is generally filed in the state where the debtor is located.
  Fact 1.2: For an individual, the state where the debtor is located is the state where the debtor resides.
  Fact 1.3: For most kinds of business organizations, the state where the debtor is located is the state of incorporation or organization.
  Fact 1.4: Many states have a state agency that operates under the secretary of state.
  Fact 1.5: That state agency is tasked with overseeing business organizations and activities, including receipt of financing statements.
  Fact 1.6: An exception exists if the collateral is tied to a particular piece of real property, such as timber, mineral rights, or fixtures.
  Fact 1.7: In that exception case, the filing must be made in the county where the property is located.
  Fact 1.8: In that exception case, the filing is usually made in the recording office or county court.
  Fact 1.9: Third parties are most likely to search for such a record in that county.

Retrieved items:
<item 1>
Ownership
A legal entity is a legal construct through which the law allows a group of natural persons to act as if it were an individual for certain purposes . Some duly
incorporated entities may not be owned by individuals nor by other entities ; they exist without being owned once they are created . Not being owned , they can
not be bought and sold . Mutual life insurance companies , credit unions , foundations and cooperatives , not for profit organizations , and public corporations
are examples of this . No person can purchase the company , as their ownership is not legally available for sale , neither as shares nor as a single whole .
Intellectual property ( edit ) Main article : Intellectual property Intellectual property ( IP ) refers to a legal entitlement which sometimes attaches to the
expressed form of an idea , or to some other intangible subject matter . This legal entitlement generally enables its holder to exercise exclusive rights of use
in relation to the subject matter of the IP . The term intellectual property reflects the idea that this subject matter is the product of the mind or the
intellect , and that IP rights may be protected at law in the same way as any other form of property .
</item 1>

<item 2>
UCC-1 financing statement
UCC - 1 financing statement - wikipedia UCC - 1 financing statement Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2014 ) ( Learn how and
when to remove this template message ) A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files
to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the
agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to
take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local
newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called ``
perfecting the security interest '' in the property , and this type of loan is a secured loan .
</item 2>

<item 3>
Bar examination
2017 ) , South Dakota , Utah , Washington , West Virginia , Wisconsin , Wyoming . The MEE can cover any of the following areas : Business associations -- Agency
and partnership , corporations , limited liability companies Conflict of laws Constitutional law Contracts Criminal law and procedure Evidence Family law
Federal civil procedure Real property Torts Trusts and estates -- decedents ' estates ; trusts and future interests Uniform Commercial Code -- Article 3 ,
Negotiable Instruments ; Article 4 ( Bank Deposits and Collections ) ; Article 9 , Secured Transactions MEE questions are actually drafted by the NCBE Drafting
Committee , with the assistance of outside academics and practitioners who are experts in the fields being tested . After initial drafting , the questions are
pretested , analyzed by outside experts and a separate NCBE committee , reviewed by boards of bar examiners in the jurisdictions that use the test , and then
revised by the Drafting Committee in accordance with the results of this process . Each MEE question is accompanied by a grading guide , and the NCBE sponsors a
grading workshop on the weekend following the bar exam whose results are provided to bar examiners .
</item 3>

<item 4>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 4>

<item 5>
United Kingdom corporation tax
Individuals could set off the tax credit against their income tax liability . On introduction , ACT was set at 30 % of the gross dividend ( the actual amount
paid plus the tax credit ) . If a company made a £ 70 dividend payment to an individual , the company would pay £ 30 of advance corporation tax . The
shareholder would receive the £ 70 cash payment , plus a tax credit of £ 30 ; thus , the individual would be deemed to have earned £ 100 , and to have already
paid tax of £ 30 on it . The ACT paid by the company would be deductible against its final `` mainstream '' corporation tax bill . To the extent that the
individual 's tax on the dividend was less than the tax credit -- for example , if his income was too low to pay tax ( below £ 595 in 1973 -- 1974 ) -- he would
be able to reclaim some or all of the £ 30 tax paid by the company .
</item 5>

<item 6>
IRS penalties
Similar penalties may apply to an income tax return ( Form 1120S ) for an S corporation return . 100 % penalty on unpaid withholding taxes ( edit ) Employers
are required to withhold income and social security taxes from wages paid to employees , and to pay these amounts promptly to the government . A penalty of 100
% of the amount not paid over ( plus liability for paying the withheld amounts ) may be collected without judicial proceedings from each and every person who
had custody and control of the funds and did not make the payment to the government . This applies to company employees and officers as individuals , as well as
to companies themselves . There have been reported cases of the IRS seizing houses of those failing to pay over employee taxes . Penalties for failure to
provide foreign information ( edit ) Taxpayers who are shareholders of controlled foreign corporations must file Form 5471 with respect to each such controlled
foreign corporation . Penalties for failure to timely file are $10,000 to $50,000 per form , plus possible loss of foreign tax credits . U.S.
</item 6>

<item 7>
UCC-1 financing statement
However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures
. In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where
third parties are most likely to search for such record . In the case of a loan secured by personal property collateral , the filing of a financing statement
gives notice of a lien against the property so that other lenders or buyers of the personal property will know of the security interest . In the case of a
filing of a financing statement by a lessor of fixtures , the filing of the financing statement gives notice of the lessor 's interests to others who acquire an
interest in the real property and related fixtures . The financing statement does not create a lien nor does it create any additional rights against a lessee in
favor of a lessor , the filing of a financing statement just gives notice of whatever rights the creditor or lessor have under their loan documents or lease ,
respectively .
</item 7>

<item 8>
IRS penalties
Jump up ^ 26 USC 6651 ( a ) ( 1 ) . Jump up ^ IRC § 6651 ( a ) ( 1 ) Jump up ^ IRC § 6651 ( a ) ( 2 ) Jump up ^ `` 1040 ( 2015 ) '' . Internal Revenue Service .
Retrieved June 28 , 2016 . Jump up ^ 26 USC 6662 . Jump up ^ 26 USC 6721 . Jump up ^ 26 CFR 301 - 6721 - 1 ( a ) ( 2 ) ( ii ) . Jump up ^ 26 USC 6698 . Jump up
^ 26 USC 6699 . Jump up ^ Technically , the Form 1120S is an income tax return , not an information return . Jump up ^ 26 USC 3101 - 3406 . Jump up ^ 26 USC
6672 . Jump up ^ 26 USC 3504 . Jump up ^ See , e.g. , United States v. St. Clair , 45 Amer . Fed . Tax Rep.
</item 8>

<item 9>
UCC-1 financing statement
References ( edit ) Jump up ^ http://www.findlegalforms.com/articles/form-encyclopedia/what-is-a-ucc1-financing-statement Jump up ^
http://biztaxlaw.about.com/od/glossaryu/g/ucc1statement.htm Retrieved from ``
https://en.wikipedia.org/w/index.php?title=UCC-1_financing_statement&oldid=803697764 '' Categories : Uniform Commercial Code Legal terminology Hidden categories
: Articles needing additional references from October 2014 All articles needing additional references Talk About Wikipedia Add links This page was last edited
on 4 October 2017 , at 02 : 00 . About Wikipedia
</item 9>

<item 10>
UCC-1 financing statement
A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of
a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit
documents and the lease , respectively , and not the financing statement . Pursuant to the standards set forth in the UCC , at 9 - 503 and 9 - 504 , the
financing statement need only contain three pieces of information : the debtor 's name and address the creditor 's name and address an indication of the
collateral , `` whether or not it is specific , if it reasonably identifies what is described . '' ( UCC 9 - 108 ) The financing statement is generally filed
with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most
kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which
is tasked with overseeing business organizations and activities , including receipt of financing statements .
</item 10>
=== END ITEM id=30c5f2ca6e467d69 ===

=== ITEM id=8e485b2b8a886355 ===
Gold sentences and facts:
Sentence 0: Their first video arcade game was 1975 's EVR Race ; a handful of others followed in the next several years , Radar Scope and Donkey Kong being among the most famous of these . The early 1980s saw Nintendo 's video game division ( led by Yokoi ) creating some of its most famous arcade titles . The massively popular Donkey Kong was created in 1981 , with Shigeru Miyamoto as its mastermind , and released in the arcades and on the Atari 2600 , Intellivision , and ColecoVision video game systems ( although Nintendo themselves generally had no involvement with these early console ports ) . This release method would be used on several later Nintendo arcade games of this same period , including the original Mario Bros. ( not to be confused with the later Super Mario Bros . ) In addition to this arcade and dedicated console game activity , Nintendo was testing the consumer handheld video game waters with the Game & Watch .
  Fact 0.0: Nintendo's first video arcade game was EVR Race, released in 1975.
  Fact 0.1: A handful of other arcade games followed Nintendo's first video arcade game in the next several years.
  Fact 0.2: Radar Scope and Donkey Kong were among the most famous of those arcade games.
  Fact 0.3: In the early 1980s, Nintendo's video game division, led by Yokoi, created some of Nintendo's most famous arcade titles.
  Fact 0.4: Donkey Kong was massively popular.
  Fact 0.5: Donkey Kong was created in 1981.
  Fact 0.6: Shigeru Miyamoto was the mastermind behind Donkey Kong.
  Fact 0.7: Donkey Kong was released in the arcades.
  Fact 0.8: Donkey Kong was released on the Atari 2600, Intellivision, and ColecoVision video game systems.
  Fact 0.9: Nintendo themselves generally had no involvement with the early console ports of Donkey Kong.
  Fact 0.10: The release method used for Donkey Kong, arcade plus console, was used on several later Nintendo arcade games of the same period.
  Fact 0.11: The original Mario Bros. was one of these later Nintendo arcade games, and it is not to be confused with the later Super Mario Bros.
  Fact 0.12: Alongside its arcade and dedicated console game activity, Nintendo was testing the consumer handheld video game market with the Game & Watch.
Sentence 1: In 1972 , the first commercially available video game console , the Magnavox Odyssey , had a light gun accessory , the Shooting Gallery . This was the first involvement of Nintendo in video games . According to Martin Picard in the International Journal of Computer Game Research : `` in 1971 , Nintendo had -- even before the marketing of the first home console in the United States -- an alliance with the American pioneer Magnavox to develop and produce optoelectronic guns for the Odyssey ( released in 1972 ) , since it was similar to what Nintendo was able to offer in the Japanese toy market in 1970s '' .
  Fact 1.0: The Magnavox Odyssey was the first commercially available video game console, and it was released in 1972.
  Fact 1.1: In 1972, the Magnavox Odyssey had a light gun accessory called the Shooting Gallery.
  Fact 1.2: The Shooting Gallery was the first involvement of Nintendo in video games.
  Fact 1.3: According to Martin Picard in the International Journal of Computer Game Research, in 1971 Nintendo had an alliance with the American pioneer Magnavox.
  Fact 1.4: The alliance was formed even before the marketing of the first home console in the United States.
  Fact 1.5: The alliance was to develop and produce optoelectronic guns for the Odyssey.
  Fact 1.6: The Odyssey was released in 1972.
  Fact 1.7: The alliance existed because Nintendo was able to offer something similar in the Japanese toy market in the 1970s.

Retrieved items:
<item 1>
History of Nintendo
Contents ( hide ) 1 Origin 2 1889 -- 1965 : As a card company 3 1966 -- 1972 : Toy company and new ventures 4 1972 -- present : Electronic games 4.1 Color TV -
Game 6 4.2 Game & Watch and Nintendo Entertainment System 4.3 Game Boy and Super NES 4.4 Nintendo 64 and Game Boy Color 4.5 Game Boy Advance and GameCube 4.6
Nintendo DS and Wii 4.7 Nintendo 3DS , Wii U , and Nintendo Switch 5 References 6 Further reading 7 External links Origin ( edit ) Nintendo poster from late
Meiji Era Nintendo started out as a small Japanese company , founded by Fusajiro Yamauchi on September 23 , 1889 , as Nintendo Koppai . Based in Kyoto , Japan ,
the business produced and marketed Hanafuda cards . The name `` Nintendo '' is commonly assumed to mean `` leave luck to heaven '' , but there are no historical
records to validate this assumption .
</item 1>

<item 2>
History of Nintendo
In 1982 , Nintendo developed a prototype system dubbed the `` Advanced Video System '' ( or AVS for short ) and had controllers much like the NES . There were
accessories such as a tape drive , a joystick and a lightgun , and along with all of that , the system was made of a computer , much like the Atari 400 ,
Commodore Vic - 20 and Commodore 64 . It was never released and is on display at the Nintendo World Store in New York . In July 1983 , Nintendo released the
Famicom ( Family Computer ) system in Japan , which was its first attempt at a cartridge - based video game console . The system sold over 500,000 units within
two months at a price around $100 USD . However , after a few months of favorable sales , Nintendo received complaints that some Famicom consoles would freeze
when the player attempted to play certain games . The fault was found in a malfunctioning chip and Nintendo decided to recall all Famicom units that were
currently on store shelves , which ultimately cost them approximately half a million USD .
</item 2>

<item 3>
History of Nintendo
The massively popular Donkey Kong was created in 1981 , with Shigeru Miyamoto as its mastermind , and released in the arcades and on the Atari 2600 ,
Intellivision , and ColecoVision video game systems ( although Nintendo themselves generally had no involvement with these early console ports ) . This release
method would be used on several later Nintendo arcade games of this same period , including the original Mario Bros. ( not to be confused with the later Super
Mario Bros . ) In addition to this arcade and dedicated console game activity , Nintendo was testing the consumer handheld video game waters with the Game &
Watch . Game & Watch and Nintendo Entertainment system ( edit ) Main articles : Game & Watch , Nintendo Entertainment System , and History of the Nintendo
Entertainment System Game & Watch or G&W is a line of handheld electronic games produced by Nintendo from 1980 to 1991 . Created by game designer Gunpei Yokoi ,
each Game & Watch features a single game to be played on an LCD screen in addition to a clock and / or an alarm . It was the earliest Nintendo product to garner
major success , with 43.4 million units sold worldwide .
</item 3>

<item 4>
History of Nintendo
History of Nintendo - wikipedia History of Nintendo Jump to : navigation , search This article needs additional citations for verification . Please help improve
this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( November 2006 ) ( Learn how and when to remove this
template message ) The history of Nintendo traces back to 1889 , when it was founded to produce handmade hanafuda cards . Nintendo Co. , Ltd. ( 任天堂 株式 会社 ,
Nintendō kabushikigaisha ) is a Japanese multinational consumer electronics company headquartered in Kyoto , Japan . It eventually became one of the most
prominent figures in today 's video game industry , being the world 's largest video game company by revenue .
</item 4>

<item 5>
History of Nintendo
These devices had a much higher novelty value than traditional toys , allowing Nintendo to charge a higher price margin for each product . Yokoi went on to
develop many other toys , including the Ten Billion Barrel puzzle , a baseball throwing machine called the Ultra Machine , and a Love Tester . Another invention
of his , in collaboration with Masayuki Uemura from Sharp , was the Nintendo Beam Gun Game , the precursor to the NES Zapper ( which was used in games like Duck
Hunt . ) . Nintendo released the first solar - powered light gun , the Nintendo Beam Gun , in 1970 ; this was the first commercially available light - gun for
home use , produced in partnership with Sharp . In 1972 , Nintendo released the Ele - Conga , one of the first programmable drum machines . It could play
pre-programmed rhythms from disc - shaped punch cards , which could be altered or programmed by the user , to play different patterns . 1972 -- present :
electronic games ( edit ) In 1972 , the first commercially available video game console , the Magnavox Odyssey , had a light gun accessory , the Shooting
Gallery . This was the first involvement of Nintendo in video games .
</item 5>

<item 6>
History of Nintendo
The first issue published was the July / August edition , which spotlighted the NES game Super Mario Bros. 2 . Nintendo Power has since ceased publication with
its December 2012 edition . Game Boy and Super NES ( edit ) Main articles : Game Boy and Super Nintendo Entertainment System The original Game Boy In 1989 ,
Nintendo ( which had seen a large amount of success from the Game & Watch ) released the Game Boy ( both created by Gunpei Yokoi ) , along with the accompanying
game Tetris . Due to the price , the game and its durability ( unlike the prior Microvision from Milton Bradley Company , which was prone to static and screen
rot ) , the Game Boy line sold extremely well , eventually amassing sales of 118 million units . Later , Super Mario Land was also released for the Game Boy ,
which sold 14 million copies worldwide . 1989 was also the year that Nintendo announced a sequel to the Famicom , to be called the Super Famicom . The last
major first - party game for the NES , Super Mario Bros. 3 , was released in early 1990 , and went on to sell over 18 million units .
</item 6>

<item 7>
Graph query row:
entity: Color TV Game 6
relationship: CREATED
related: Nintendo
date: 1977
kind: video game system
fact: None
context: None
role: None
relationship_detail: None
relation_detail: None
</item 7>

<item 8>
Graph query row:
entity: EVR Race
relationship: CREATED
related: Nintendo
date: 1975
kind: video arcade game
fact: None
context: None
role: None
relationship_detail: None
relation_detail: None
</item 8>

<item 9>
History of Nintendo
Color TV - game 6 ( edit ) Main article : Color TV Game Nintendo 's first video game system , the Color TV Game 6 Nintendo at this time saw how successful video
games were and began to dabble in them . Their first step in that field was to secure the rights to distribute the Magnavox Odyssey in Japan in 1975 . At the
time , home video game consoles were extremely rare -- even the seminal Atari Pong console had yet to be produced . After experiencing reasonable success at
this , Nintendo began developing its own video games , both for home and for arcades . In the 1970s , Mitsubishi Electric proposed joint development of `` Color
TV Game Machine '' . In 1977 , they released `` Color TV Game 6 '' and `` Color TV Game 15 '' ( 6 and 15 indicates the number of games ) . Their first video
arcade game was 1975 's EVR Race ; a handful of others followed in the next several years , Radar Scope and Donkey Kong being among the most famous of these .
The early 1980s saw Nintendo 's video game division ( led by Yokoi ) creating some of its most famous arcade titles .
</item 9>

<item 10>
History of Nintendo
This was the first involvement of Nintendo in video games . According to Martin Picard in the International Journal of Computer Game Research : `` in 1971 ,
Nintendo had -- even before the marketing of the first home console in the United States -- an alliance with the American pioneer Magnavox to develop and
produce optoelectronic guns for the Odyssey ( released in 1972 ) , since it was similar to what Nintendo was able to offer in the Japanese toy market in 1970s
'' . In 1973 , its focus shifted to family entertainment venues with the Laser Clay Shooting System , using the same light gun technology used in Nintendo 's
Kousenjuu series of toys , and set up in abandoned bowling alleys . Following some success , Nintendo developed several more light gun machines for the emerging
arcade scene . While the Laser Clay Shooting System ranges had to be shut down following excessive costs , Nintendo had found a new market . In 1974 , Nintendo
secured the rights to distribute the Magnavox Odyssey video game console in Japan .
</item 10>
=== END ITEM id=8e485b2b8a886355 ===

=== ITEM id=a601e40e851b0758 ===
Gold sentences and facts:
Sentence 0: Health or vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually rewarded in some way .
  Fact 0.0: Health or vitality is an attribute assigned to entities in a role-playing or video game.
  Fact 0.1: Health or vitality is assigned to entities such as the player character, enemies, and objects.
  Fact 0.2: Health or vitality indicates the state of an entity in combat.
  Fact 0.3: Health is usually measured in hit points or health points.
  Fact 0.4: Hit points or health points are shortened to HP.
  Fact 0.5: When the HP of a player character reaches zero, the player may lose a life.
  Fact 0.6: When the HP of a player character reaches zero, the character might become incapacitated or die.
  Fact 0.7: When the HP of an enemy reaches zero, the enemy may be defeated or die.
  Fact 0.8: When the HP of an enemy reaches zero, the player is usually rewarded in some way.

Retrieved items:
<item 1>
Health (gaming)
`` Falcom : Legacy of Ys '' . Games ( 111 ) : 152 -- 159 ( 153 ) . ( cf . Szczepaniak , John ( July 8 , 2011 ) . `` History of Ys interviews '' . Hardcore
Gaming 101 . Retrieved 6 September 2011. ) Jump up ^ Jon Martindale ( 2012 - 10 - 03 ) . `` Let 's Kill off Health Bars '' . Kit Guru Gaming . Archived from the
original on May 28 , 2015 . Retrieved 2014 - 11 - 21 . Jump up ^ Novak , Jeannie ( 2013 - 04 - 11 ) . The Official GameSalad Guide to Game Development . Cengage
Learning . p. 31 . ISBN 1133605648 . Retrieved 2014 - 11 - 21 . Jump up ^ Voorhees , Gerald A. ; Call , Joshua ; Whitlock , Katie ( 2012 - 11 - 02 ) .
</item 1>

<item 2>
Health (gaming)
`` Disposable Bodies : Cyborg Regeneration and FPS Mechanics '' . Guns , Grenades , and Grunts : First - Person Shooter Games ( Google eBook ) . Bloomsbury
Publishing USA . ISBN 1441191445 . Retrieved 2014 - 11 - 21 . ( hide ) Concepts of video games Glossary of video game terms Attributes Health Life Experience
point Magic Mechanics Item Power - up HUD Warp Fog of war Invisible wall Cutscene Mini-map Paper doll Destructible environment Loading screen Replay value
Scripted sequence Password Line of sight Critical hit Level Permadeath Status effect Hitbox Movement techniques Rocket jumping Strafe - jumping Strafing
Trickjump Modes Player type Single - player Multiplayer Cooperative Competitive gameplay eSports Speedrun Gameplay type Capture the flag ( CTF ) Deathmatch Last
man standing Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Health (gaming)
US Navy used similar concept in their tactical war games already in 1920s and 1930s . In their simulation each ship had Life parameter . The unit of Life of the
ship was a number of `` equivalent penetrative 14 - inch shell hits '' . The Navy considered ie. that Kongō - class battlecruiser had 12 Life points and Nagato
- class battleship had 18.8 . A visual power meter representing stamina was used in Nintendo 's 1983 arcade game Punch - Out ! ! and Data East 's 1981 DECO
Cassette System arcade game Flash Boy . Usage ( edit ) In action video games as well as in role - playing games , health points can usually be depleted by
attacking the entity . A defense attribute might reduce the amount of HP that is lost when a character is damaged . It is common in role - playing games for a
character 's maximum health and defense attributes to be gradually raised as the character levels up . In game design , it is deemed important that a player is
aware of it when they are losing health , each hit playing a clear sound effect .
</item 3>

<item 4>
Health (gaming)
This mechanic initially appeared in action role - playing games , with early examples including the Hydlide series , the Ys series , and Woody Poco . In Woody
Poco , the rate at which health recharges is based on food level . In Hydlide and Ys , the player character has to stand still for their health to automatically
regenerate . This system was popularized in first - person shooters by Halo : Combat Evolved ( 2001 ) , though regenerating health in The Getaway ( 2002 ) has
been cited to be more comparable to later use of the mechanic in first - person shooters . Display ( edit ) Heart - shaped icons can indicate the amount of
health a player has left . The way health is displayed on the screen has an effect on the player . Many games only show the health of the player character ,
while keeping the health of enemies hidden . This is done in the Legend of Zelda series , Minecraft and Monster Hunter series to keep the player 's progress in
defeating their enemy unclear and therefore exciting . In these games , the fact that the enemies are being damaged is indicated by their behavior .
</item 4>

<item 5>
Health (gaming)
Health ( Gaming ) - wikipedia Health ( Gaming ) Jump to : navigation , search A health bar , a possible representation of the health of a character . Health or
vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its
state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may
lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually
rewarded in some way . Any entity within a game could have a health value , including the player character , non-player characters and objects . Indestructible
entities have no diminishable health value . Health might be displayed as a numeric value , such as `` 50 / 100 '' . Here , the first number indicates the
current amount of HP an entity has and the second number indicates the entity 's maximum HP .
</item 5>

<item 6>
Health (gaming)
In these games , the fact that the enemies are being damaged is indicated by their behavior . On the other hand , many fighting games , such as the Street
Fighter series , use easy - to - read health bars to clearly indicate the progress the player is making with each hit . It is common in first - person shooters
to indicate low health of the player character by blood spatters or by a distorted red hue on the screen , attempting to mimic the effects of wounding and
trauma . These visual effects fade as health regenerates . References ( edit ) ^ Jump up to : Moore , Michael ( 2011 - 03 - 23 ) . Basics of Game Design . CRC
Press . pp. 151 , 194 . ISBN 1439867763 . Retrieved 2014 - 12 - 09 . Jump up ^ Chris Antista . `` The 10 most creative life bars '' . GamesRadar. 2010 - 08 - 17
. p. 2 . Archived from the original on 2014 - 12 - 28 .
</item 6>

<item 7>
Health (gaming)
In video games , health can also be displayed graphically , such as with a bar that empties itself when an entity loses health ( a health bar , typically red )
, icons that are `` chipped away '' , or in more novel ways . Contents ( hide ) 1 History 2 Usage 2.1 Regeneration 3 Display 4 References History ( edit )
Dungeons & Dragons co-creator Dave Arneson described the origin of hit points in a 2002 interview . When Arneson was adapting the medieval wargame Chainmail (
1971 ) to a fantasy setting , a process that with Gary Gygax would lead to Dungeons & Dragons , he saw that the emphasis of the gameplay was moving from large
armies to small groups of heroes and eventually to the identification of one player and one character that is essential to role - playing as it was originally
conceived . Players became attached to their heroes and did not want them to die every time they lost a die roll . Players were thus given multiple hit points
which were incrementally decreased as they took damage . Arneson took the concept , along with armor class , from a set of a naval American Civil War game 's
rules .
</item 7>

<item 8>
Health (gaming)
wikipedia.org/w/index.php?title=Health_(gaming)&oldid=819315199 '' Categories : Role - playing game terminology Video game terminology Hidden categories : All
articles lacking reliable references Articles lacking reliable references from September 2016 Talk Contents About Wikipedia Deutsch Español فارسی Français 한국어
Italiano עברית ქართული Nederlands 日本 語 Norsk Norsk nynorsk Polski Português Русский Suomi Svenska Українська 中文 Edit links This page was last edited on 8
January 2018 , at 18 : 11 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site ,
you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 8>

<item 9>
Health (gaming)
Author Scott Rogers states that `` health should deplete in an obvious manner , because with every hit , a player is closer to losing their life . '' The
display of health also helps to dramatize the near - loss of a life . Regeneration ( edit ) Player characters can often restore their health points by consuming
certain items , such as health potions , food or first - aid kits . Staying a night at an inn fully restores a character 's health in many role - playing video
games . In general , the different methods of regenerating health has its uses in a particular genre . In action games , this method is very quick , whereas
role - playing games feature slower paced methods to match the gameplay and realism . Some video games feature automatically regenerating health , where lost
health points are regained over time . This can be useful to not `` cripple '' the player , making them still able to continue even after losing lots of health
. However , automatically regenerating health may also cause a player to `` power through '' sections they might otherwise have had to approach cautiously ,
simply because there are no lasting consequences to losing a large amount of health .
</item 9>

<item 10>
Health (gaming)
Jump up ^ Jonathan Moriarty ( 2010 - 12 - 02 ) . `` Video Game Basics : The Health Bar '' . Baltimoregamer.com . Archived from the original on 28 April 2012 .
Retrieved 2014 - 11 - 21 . ^ Jump up to : Dunn , Jeff ( 2012 - 11 - 15 ) . `` Stop , Drop , and Heal : The history of regenerating health '' . GamesRadar .
Retrieved 2015 - 01 - 08 . Jump up ^ Sulliven , Lucas . `` Top 7 ... Games you did n't know did it first '' . GamesRadar . Retrieved 2015 - 01 - 08 . ^ Jump up
to : John Szczepaniak ( 2016 ) . `` dB - SOFT Gaming 101 '' . The Untold History of Japanese Game Developers . 2 . Jump up ^ Szczepaniak , John ( 7 July 2011 )
. `` Falcom : Legacy of Ys '' .
</item 10>
=== END ITEM id=a601e40e851b0758 ===
