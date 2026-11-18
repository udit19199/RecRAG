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

=== ITEM id=d7e8fd8f54e3895e ===
Gold sentences and facts:
Sentence 0: Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C. , bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming the longest serving moderator in the program 's history .
  Fact 0.0: Network officials were concerned for the show's future.
  Fact 0.1: Network officials turned to Tim Russert.
  Fact 0.2: Tim Russert was the network's Washington, D.C., bureau chief.
  Fact 0.3: Tim Russert took over as moderator of Meet the Press on December 8, 1991.
  Fact 0.4: Tim Russert remained with Meet the Press until his death on June 13, 2008.
  Fact 0.5: Tim Russert became the longest serving moderator in the history of Meet the Press.
Sentence 1: Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966 -- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley 1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present
  Fact 1.0: Martha Rountree was moderator from 1947 to 1953.
  Fact 1.1: Ned Brooks was moderator from 1953 to 1965.
  Fact 1.2: Lawrence E. Spivak was moderator from 1966 to 1975.
  Fact 1.3: Bill Monroe was moderator from 1975 to 1984.
  Fact 1.4: Roger Mudd and Marvin Kalb were co-moderators from 1984 to 1985.
  Fact 1.5: Marvin Kalb was moderator from 1985 to 1987.
  Fact 1.6: Chris Wallace was moderator from 1987 to 1988.
  Fact 1.7: Garrick Utley was moderator from 1989 to 1991.
  Fact 1.8: Tim Russert was moderator from 1991 to 2008.
  Fact 1.9: Tom Brokaw was moderator in 2008.
  Fact 1.10: David Gregory was moderator from 2008 to 2014.
  Fact 1.11: Chuck Todd has been moderator since 2014.

Retrieved items:
<item 1>
Meet the Press
The next week , Bill Monroe , previously a weekly panelist like Spivak had been years before , took over as moderator and stayed until June 2 , 1984 . For the
next seven and a half years , the program then went through a series of hosts as it struggled in the ratings against ABC 's This Week with David Brinkley .
Roger Mudd and Marvin Kalb , as co-moderators , followed Monroe for a year , followed by Chris Wallace ( who would later to go on to a much longer run as host
of the rival program Fox News Sunday ) from 1987 to 1988 . Garrick Utley , then hosting Weekend Today , concurrently hosted Meet the Press from 1989 through
December 1 , 1991 . All this occurred despite the increasing ratings of NBC News ' other programs ( and those of the network generally ) during that period .
The program originally aired at noon Eastern Time every Sunday , before moving to a 9 : 00 a.m. slot by the early 1990s . Under Russert ( edit ) Meet the Press
logo used from September 10 , 1995 to June 2008 .
</item 1>

<item 2>
Meet the Press
It originates from NBC 's bureau in Washington , D.C. The longevity of Meet the Press can be illustrated in consideration of the fact that the program debuted
during what was only the second official `` network television season '' for American television . One historical landmark of the program is that it was the
first on which a sitting U.S. President , Gerald Ford , appeared on a live television network news program , which occurred on the November 9 , 1975 broadcast .
The program has been hosted by 12 different moderators to date , beginning with creator Martha Rountree . The show 's moderator since 2014 is Chuck Todd , who
also serves as political director for NBC News . Currently , the hour - long program airs in most markets on Sundays at 9 : 00 a.m. live in the Eastern Time
Zone and on tape delay elsewhere . Meet the Press is also occasionally pre-empted due to network coverage of sports events held outside of the U.S. The program
is also rebroadcast on Sundays at 2 : 00 p.m. , and Mondays at 2 : 00 a.m. and sometimes 4 : 00 a.m.
</item 2>

<item 3>
Meet the Press
The following week on August 17 , he moderated the entire broadcast . It was also reported on December 1 , 2008 , that the December 7 broadcast would be Brokaw
's last , with Gregory becoming the new permanent host the following Sunday . Under Gregory ( edit ) David Gregory began his tenure as moderator on December 14
, 2008 . Four days after Gregory 's first regular broadcast , on December 18 , NBC News political director Chuck Todd was named contributing editor of Meet the
Press . Throughout Gregory 's tenure as moderator , Meet the Press experienced significant ratings declines . In the final three months of 2013 , the program
placed third among the Sunday morning talk shows in total viewership , behind CBS 's Face the Nation and ABC 's This Week for the first time since 1992 ; it
also experienced the lowest ratings in the show 's entire history among the key 25 - to - 54 age viewing demographic during this period . NBC management became
uncertain as to the future direction of the program .
</item 3>

<item 4>
Meet the Press
Russert interviews General Peter Pace in 2006 . Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C.
, bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming
the longest serving moderator in the program 's history . Under Russert , the program was expanded to one hour and became less of a televised press conference ,
focusing more on Russert 's questions and comments ; Russert also engaged in longer in - depth interviews and hosted panels of experts to discuss the topics
featured in that week 's broadcast . Russert signed off each edition by saying , `` That 's all for today . We 'll be back next week . If it 's Sunday , it 's
Meet the Press . '' During the professional football season , Russert , a native of Buffalo , New York , and an avid fan of the Buffalo Bills , sometimes added
, `` Go Bills ! , '' and occasionally would ask panelists , `` How ' bout those Sabres ? ''
</item 4>

<item 5>
Meet the Press
Meet the Press - Wikipedia Meet the Press Jump to : navigation , search This article is about the American public affairs program . For the Australian program ,
see Meet the Press ( Australian TV program ) . Meet the Press Genre Public affairs / news analysis program Created by Martha Rountree Lawrence E. Spivak
Directed by Rob Melick Presented by Chuck Todd ( for past moderators , see section ) Narrated by Fred Facey Bert Pence Dennis Haysbert Theme music composer John
Williams Opening theme `` The Pulse of Events '' ( fourth part of The Mission ) Country of origin United States Original language ( s ) English No. of seasons
70 No. of episodes 3,600 + ( weekly ) ( as of 2017 ) Production Executive producer ( s ) John Reiss Location ( s ) NBC News Washington Bureau , Washington ,
D.C.
</item 5>

<item 6>
Meet the Press
This section , in the next two paragraphs , needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) Meet the Press was
originally presented as a 30 - minute press conference with a single guest and a panel of questioners . Its first guest was James Farley , who served as
Postmaster General , Democratic National Committee chairman and campaign manager to Franklin Delano Roosevelt under the first two terms of the New Deal
Administration . Creator Rountree served as its first host , the program 's only female moderator to date . She stepped down on November 1 , 1953 , and was
succeeded by Ned Brooks , who remained as moderator until his retirement on December 26 , 1965 . Spivak became the moderator on January 1 , 1966 , moving up
from his role as a permanent panelist . He retired on November 9 , 1975 , on a special one - hour edition that featured , for the first time , a sitting
president , Gerald Ford , as guest .
</item 6>

<item 7>
Meet the Press
On May 2 , 2010 , Meet the Press became the last NBC News program to convert to high definition , and unveiled a new set consisting of large video screens
mostly used to display Washington scenery , satellite interview subjects and moderator and subject talking points , along with graphics produced for the format
. Moderators ( edit ) The following is the list of moderators for Meet the Press : Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966
-- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley
1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present On - remote broadcasts ( outside D.C. studios ) (
edit ) Date Location Event / Reason July 17 , 1988 Atlanta , Georgia 1988 Democratic National Convention August 14 , 1988 New Orleans ,
</item 7>

<item 8>
Meet the Press
Bureau Chief and was given `` executive oversight '' of Meet the Press . Interim Brokaw era ( edit ) NBC Nightly News anchor Brian Williams acted as moderator
of the first show following the June 15 tribute to Russert , with the same guests and subject matter that Russert was planning for when he died . Following
Russert 's death , Tom Brokaw was named the interim moderator through the 2008 general elections . Brokaw followed Russert 's tradition by signing off with ``
We 'll be back next Sunday because if it 's Sunday , it 's Meet the Press '' ( a sign - off that would continue to be used by his successors as moderator ) . In
September of that year , the show was presented with limited commercial interruption . On August 10 , David Gregory moderated the panel discussion during the
second half - hour of the broadcast , while Brokaw anchored the first half - hour from the site of the Summer Olympics in Beijing . The following week on August
17 , he moderated the entire broadcast .
</item 8>

<item 9>
Meet the Press
studios ) 3.2 Notable guests and events 4 Distribution 5 See also 6 References and footnotes 7 External links Format ( edit ) The program 's format consists of
an extended one - on - one interview with the host , and is sometimes followed by a roundtable discussion or one - on - two interview with figures in
adversarial positions , either Congressional members from opposite sides of the aisle or political commentators . Originally a half - hour program for most of
its history , the show expanded to 60 minutes starting with the September 20 , 1992 , broadcast . The program also features in - depth examinations of facts
behind political and general news stories ( particularly as part of a segment called the `` Data Download '' , introduced after Chuck Todd assumed duties as
moderator , which is conducted on a touchscreen within the main set . History ( edit ) Meet the Press set , November 1975 . On this broadcast , a sitting
American president ( Gerald Ford ) was , for the first time , a guest on a live television network news program .
</item 9>

<item 10>
Meet the Press
Camera setup Multi-camera Running time 30 minutes ( 1947 -- 1992 ) 52 minutes ( 1992 -- present ) Production company ( s ) NBC News Productions Release Original
network NBC Picture format 480i ( 4 : 3 SDTV ) ( 1947 -- 2010 ) 720p ( 4 : 3 SDTV ) ( 1996 -- 2010 ) 1080i ( 16 : 9 HDTV ) ( 2010 -- present ) Original release
November 6 , 1947 ( 1947 - 11 - 06 ) -- present External links Website www.nbcnews.com/meet-the-press Meet the Press is a weekly American television news /
interview program broadcast on NBC . It is the longest - running program in television history , though the current format bears little resemblance to the debut
episode on November 6 , 1947 . Meet the Press specializes in interviews with leaders in Washington , D.C. , across the country and even the world on issues of
politics , economics , foreign policy and other public affairs , along with panel discussions that provide opinions and analysis . It originates from NBC 's
bureau in Washington , D.C.
</item 10>
=== END ITEM id=d7e8fd8f54e3895e ===

=== ITEM id=39b8158714e2ec67 ===
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
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 1>

<item 2>
IRS penalties
Willis , Hoffman , et al , South - Western Federal Taxation , 2009 Edition ISBN 978 - 0 - 324 - 66052 - 4 , pages 26 -- 13 through 26 - 19 . References ( edit )
Jump up ^ Most of the penalty rules are in Chapters 68 through 75 of the Internal Revenue Code , sections 6651 through 7344 . 26 USC Subtitle F . Jump up ^ Alan
J. Tarr , J.D. , LL. M. ( Taxation ) & Pamela Jensen Drucker , J.D. , LL. M. ( Taxation ) , Civil Tax Penalties , U.S. Income Portfolios , Vol. 634 ( 2d ed.
2012 ) , Bloomberg BNA . Jump up ^ 26 USC 6654 . Jump up ^ 26 USC 6651 ( a ) ( 3 ) . Jump up ^ See IRS Tax Topic 653 . Jump up ^ 26 USC 6651 ( a ) ( 1 ) .
</item 2>

<item 3>
IRS penalties
This penalty of 20 % or 40 % of the increase in tax is due in the case of substantial understatement of tax , substantial valuation misstatements , transfer
pricing adjustments , or negligence or disregard of rules or regulations . For example , a valuation overstatement can result in a 30 % penalty on the amount of
tax owed . Special rules apply for each of these types of errors under which the penalty may be waived . Penalties in connection with information returns ( edit
) Certain types of returns , called `` information returns , '' do not require payment of tax . These include forms filed by employers to report wages ( Form W
- 2 ) and businesses to report certain payments ( Form 1099 series instructions ) . The penalty for failures related to these forms is a dollar amount per form
not timely filed , and the amount of penalty increases with the degree of lateness . The current maximum penalty for these forms is $50 . Many of the forms must
be filed electronically , and filing on paper is considered non-filing . Late filing of returns of partnership income ( Form 1065 ) can result in penalties of
$195 per month per partner , up to a maximum of 12 months .
</item 3>

<item 4>
IRS penalties
Similar penalties may apply to an income tax return ( Form 1120S ) for an S corporation return . 100 % penalty on unpaid withholding taxes ( edit ) Employers
are required to withhold income and social security taxes from wages paid to employees , and to pay these amounts promptly to the government . A penalty of 100
% of the amount not paid over ( plus liability for paying the withheld amounts ) may be collected without judicial proceedings from each and every person who
had custody and control of the funds and did not make the payment to the government . This applies to company employees and officers as individuals , as well as
to companies themselves . There have been reported cases of the IRS seizing houses of those failing to pay over employee taxes . Penalties for failure to
provide foreign information ( edit ) Taxpayers who are shareholders of controlled foreign corporations must file Form 5471 with respect to each such controlled
foreign corporation . Penalties for failure to timely file are $10,000 to $50,000 per form , plus possible loss of foreign tax credits . U.S.
</item 4>

<item 5>
IRS penalties
Jump up ^ 26 USC 6651 ( a ) ( 1 ) . Jump up ^ IRC § 6651 ( a ) ( 1 ) Jump up ^ IRC § 6651 ( a ) ( 2 ) Jump up ^ `` 1040 ( 2015 ) '' . Internal Revenue Service .
Retrieved June 28 , 2016 . Jump up ^ 26 USC 6662 . Jump up ^ 26 USC 6721 . Jump up ^ 26 CFR 301 - 6721 - 1 ( a ) ( 2 ) ( ii ) . Jump up ^ 26 USC 6698 . Jump up
^ 26 USC 6699 . Jump up ^ Technically , the Form 1120S is an income tax return , not an information return . Jump up ^ 26 USC 3101 - 3406 . Jump up ^ 26 USC
6672 . Jump up ^ 26 USC 3504 . Jump up ^ See , e.g. , United States v. St. Clair , 45 Amer . Fed . Tax Rep.
</item 5>

<item 6>
IRS penalties
Tax adviser penalties ( edit ) Penalties also apply to people who promote tax shelters or who fail to maintain and disclose lists of reportable transactions
their customers or clients for those transactions . These monetary penalties can be severe . Judicial appeal of penalties ( edit ) Most penalties are subject to
judicial review . However , the courts rarely modify assessment of the penalties and interest for underestimate or late payment . Further reading ( edit )
Meldman , Robert E. , Federal Taxation Practice and Procedure , ISBN 978 - 0 - 8080 - 1647 - 2 , Chapters 9 and 18 Shafiroff , Ira L. , Internal Revenue Service
Practice and Procedure Deskbook , Kindle edition ( no ISBN ) , or hardcover ISBN 978 - 0 - 87224 - 109 - 1 Pratt , Kulsrud , et al , Federal Taxation , 2006
Edition ISBN 978 - 0 - 7593 - 5175 - 2 , pages 2 -- 6 through 2 - 15 .
</item 6>

<item 7>
IRS penalties
Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines . Contents ( hide ) 1 Underestimate and late payment
penalties 2 Penalties for failure to timely file returns or timely pay tax 3 Accuracy related penalties 4 Penalties in connection with information returns 5 100
% penalty on unpaid withholding taxes 6 Penalties for failure to provide foreign information 7 Excise taxes as penalties 8 Tax fraud penalties 9 Tax adviser
penalties 10 Judicial appeal of penalties 11 Further reading 12 References Underestimate and late payment penalties ( edit ) Taxpayers are required to have
withholding of tax or make quarterly estimated tax payments before the end of the tax year . Because accurate estimation requires accurate prediction of the
future , taxpayers sometimes underestimate the amount due . The penalty for paying too little estimated tax or having too little tax withheld is computed with
interest on the amount that should have been , but was not , paid . Where a taxpayer has filed an income or excise tax return that shows a balance due but does
not pay that balance by the due date of the return ( without extensions ) , a different charge applies .
</item 7>

<item 8>
IRS penalties
IRS penalties - wikipedia IRS penalties Jump to : navigation , search This article is part of a series on Taxation in the United States of America Federal
taxation ( show ) Alternative minimum tax Capital gains tax Corporate tax Estate tax Excise tax Gift tax Income tax Payroll tax Internal Revenue Service ( IRS )
Internal Revenue Code ( IRC ) IRS tax forms Revenue by state History Constitutional authority Taxpayer standing Court Protest Evasion State and local taxation (
show ) State income tax Property tax Sales tax Use tax Land value tax State tax levels Federal tax reform ( show ) Automated payment transaction tax 9 -- 9 -- 9
Competitive Tax Plan Efficient Taxation of Income FairTax Flat tax Hall -- Rabushka flat tax Kemp Commission Taxpayer Choice Act USA Tax Value added tax Border
- adjustment tax United States portal Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters
. The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties
specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
</item 8>

<item 9>
IRS penalties
Penalties may be monetary or may involve forfeiture of property . Criminal penalties may include jail time , but are imposed only by a federal judge after a
defendant is convicted . Most monetary penalties are based on the amount of tax not properly paid . Penalties may increase with the period of nonpayment . Some
penalties are fixed dollar amounts or fixed percentages of some measure required to be reported . Excise taxes used as penalties are imposed in the Code
sections relating to particular kinds of transactions . Some penalties may be waived or abated where the taxpayer shows reasonable cause for the failure .
Penalties apply for failures to file income tax returns or information returns , or for filing incorrect returns . Some penalties may be very minor . Penalties
apply for certain types of errors on tax returns , and may be substantial . Some penalties are imposed as excise taxes on particular transactions . Certain
other penalties apply for other types of failures . Certain acts may result in forfeiture of property of the taxpayer . There are over 150 kinds of civil
penalties in the U.S. Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines .
</item 9>

<item 10>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 10>
=== END ITEM id=39b8158714e2ec67 ===

=== ITEM id=46e51e8e9350490f ===
Gold sentences and facts:
Sentence 0: ( hide ) Climate data for The South Pole Month Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Year Record high ° C ( ° F ) − 14.4 ( 6.1 ) − 20.6 ( − 5.1 ) − 26.7 ( − 16.1 ) − 27.8 ( − 18 ) − 25.1 ( − 13.2 ) − 28.8 ( − 19.8 ) − 33.9 ( − 29 ) − 32.8 ( − 27 ) − 29.3 ( − 20.7 ) − 25.1 ( − 13.2 ) − 18.9 ( − 2 ) − 12.3 ( 9.9 ) − 12.3 ( 9.9 ) Average high ° C ( ° F ) − 26.0 ( − 14.8 ) − 37.9 ( − 36.2 ) − 49.6 ( − 57.3 ) − 53.0 ( − 63.4 ) − 53.6 ( − 64.5 ) − 54.5 ( − 66.1 ) − 55.2 ( − 67.4 ) − 54.9 ( − 66.8 ) − 54.4 ( − 65.9 ) − 48.4 ( − 55.1 ) − 36.2 ( − 33.2 ) − 26.3 ( − 15.3 ) − 45.8 ( − 50.4 ) Daily mean ° C ( ° F ) − 28.4 ( − 19.1 ) − 40.9 ( − 41.6 ) − 53.7 ( − 64.7 ) − 57.8 ( − 72 ) − 58.0 ( − 72.4 ) − 58.9 ( − 74 ) − 59.8 ( − 75.6 ) − 59.7 ( − 75.5 ) − 59.1 ( − 74.4 ) − 51.6 ( − 60.9 ) − 38.2 ( − 36.8 ) − 28.0 ( − 18.4 ) − 49.5 ( − 57.1 ) Average low ° C ( ° F ) − 29.6 ( − 21.3 ) − 43.1 ( − 45.6 ) − 56.8 ( − 70.2 ) − 60.9 ( − 77.6 ) − 61.5 ( − 78.7 ) − 62.8 ( − 81 ) − 63.4 ( − 82.1 ) − 63.2 ( − 81.8 ) − 61.7 ( − 79.1 ) − 54.3 ( − 65.7 ) − 40.1 ( − 40.2 ) − 29.1 ( − 20.4 ) − 52.2 ( − 62 ) Record low ° C ( ° F ) − 41.1 ( − 42 ) − 58.9 ( − 74 ) − 71.1 ( − 96 ) − 75.0 ( − 103 ) − 78.3 ( − 108.9 ) − 82.8 ( − 117 ) − 80.6 ( − 113.1 ) − 79.3 ( − 110.7 ) − 79.4 ( − 110.9 ) − 72.0 ( − 97.6 ) − 55.0 ( − 67 ) − 41.1 ( − 42 ) − 82.8 ( − 117 ) Average precipitation mm ( inches ) 0.3 ( 0.012 ) 0.6 ( 0.024 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) Trace Trace 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.3 ( 0.012 ) 2.3 ( 0.091 ) Average precipitation days ( ≥ 0.1 mm ) 0.2 0.3 0.2 0.0 0.2 0.1 0.0 0.0 0.1 0.1 0.1 0.3 1.6 Average snowy days 22.0 19.6 13.6 11.4 17.2 17.3 18.2 17.5 11.7 16.7 16.9 20.6 203.0 Mean monthly sunshine hours 406.1 497.2 195.3 0.0 0.0 0.0 0.0 0.0 34.1 390.6 558.0 616.9 2,698.2 Mean daily sunshine hours 13.1 17.6 6.3 0.0 0.0 0.0 0.0 0.0 1.1 12.6 18.6 19.9 7.4 Source # 1 : Pogoda.ru.net ( temperatures , 1981 -- 2010 , extremes 1957 -- present ) Source # 2 : Deutscher Wetterdienst ( precipitation 1957 -- 1988 and sun 1978 -- 1993 ) , NOAA ( snowy days data , 1961 -- 1988 )
  Fact 0.0: The climate data table is for The South Pole.
  Fact 0.1: The record high temperatures at The South Pole in degrees Celsius (Fahrenheit) are -14.4 (6.1) in January, -20.6 (-5.1) in February, -26.7 (-16.1) in March, -27.8 (-18) in April, -25.1 (-13.2) in May, -28.8 (-19.8) in June, -33.9 (-29) in July, -32.8 (-27) in August, -29.3 (-20.7) in September, -25.1 (-13.2) in October, -18.9 (-2) in November and -12.3 (9.9) in December.
  Fact 0.2: The record high temperature for the year at The South Pole is -12.3 (9.9) degrees Celsius (Fahrenheit).
  Fact 0.3: The average high temperatures at The South Pole in degrees Celsius (Fahrenheit) are -26.0 (-14.8) in January, -37.9 (-36.2) in February, -49.6 (-57.3) in March, -53.0 (-63.4) in April, -53.6 (-64.5) in May, -54.5 (-66.1) in June, -55.2 (-67.4) in July, -54.9 (-66.8) in August, -54.4 (-65.9) in September, -48.4 (-55.1) in October, -36.2 (-33.2) in November and -26.3 (-15.3) in December.
  Fact 0.4: The average high temperature for the year at The South Pole is -45.8 (-50.4) degrees Celsius (Fahrenheit).
  Fact 0.5: The daily mean temperatures at The South Pole in degrees Celsius (Fahrenheit) are -28.4 (-19.1) in January, -40.9 (-41.6) in February, -53.7 (-64.7) in March, -57.8 (-72) in April, -58.0 (-72.4) in May, -58.9 (-74) in June, -59.8 (-75.6) in July, -59.7 (-75.5) in August, -59.1 (-74.4) in September, -51.6 (-60.9) in October, -38.2 (-36.8) in November and -28.0 (-18.4) in December.
  Fact 0.6: The daily mean temperature for the year at The South Pole is -49.5 (-57.1) degrees Celsius (Fahrenheit).
  Fact 0.7: The average low temperatures at The South Pole in degrees Celsius (Fahrenheit) are -29.6 (-21.3) in January, -43.1 (-45.6) in February, -56.8 (-70.2) in March, -60.9 (-77.6) in April, -61.5 (-78.7) in May, -62.8 (-81) in June, -63.4 (-82.1) in July, -63.2 (-81.8) in August, -61.7 (-79.1) in September, -54.3 (-65.7) in October, -40.1 (-40.2) in November and -29.1 (-20.4) in December.
  Fact 0.8: The average low temperature for the year at The South Pole is -52.2 (-62) degrees Celsius (Fahrenheit).
  Fact 0.9: The record low temperatures at The South Pole in degrees Celsius (Fahrenheit) are -41.1 (-42) in January, -58.9 (-74) in February, -71.1 (-96) in March, -75.0 (-103) in April, -78.3 (-108.9) in May, -82.8 (-117) in June, -80.6 (-113.1) in July, -79.3 (-110.7) in August, -79.4 (-110.9) in September, -72.0 (-97.6) in October, -55.0 (-67) in November and -41.1 (-42) in December.
  Fact 0.10: The record low temperature for the year at The South Pole is -82.8 (-117) degrees Celsius (Fahrenheit).
  Fact 0.11: The average precipitation at The South Pole in millimeters (inches) is 0.3 (0.012) in January, 0.6 (0.024) in February, 0.2 (0.008) in March, 0.1 (0.004) in April, 0.2 (0.008) in May, 0.1 (0.004) in June, Trace in July, Trace in August, 0.1 (0.004) in September, 0.1 (0.004) in October, 0.1 (0.004) in November and 0.3 (0.012) in December.
  Fact 0.12: The average precipitation for the year at The South Pole is 2.3 (0.091) millimeters (inches).
  Fact 0.13: The average number of days with precipitation of at least 0.1 mm at The South Pole is 0.2 in January, 0.3 in February, 0.2 in March, 0.0 in April, 0.2 in May, 0.1 in June, 0.0 in July, 0.0 in August, 0.1 in September, 0.1 in October, 0.1 in November and 0.3 in December.
  Fact 0.14: The average number of days with precipitation of at least 0.1 mm at The South Pole is 1.6 for the year.
  Fact 0.15: The average number of snowy days at The South Pole is 22.0 in January, 19.6 in February, 13.6 in March, 11.4 in April, 17.2 in May, 17.3 in June, 18.2 in July, 17.5 in August, 11.7 in September, 16.7 in October, 16.9 in November and 20.6 in December.
  Fact 0.16: The average number of snowy days at The South Pole is 203.0 for the year.
  Fact 0.17: The mean monthly sunshine hours at The South Pole are 406.1 in January, 497.2 in February, 195.3 in March, 0.0 in April, 0.0 in May, 0.0 in June, 0.0 in July, 0.0 in August, 34.1 in September, 390.6 in October, 558.0 in November and 616.9 in December.
  Fact 0.18: The mean monthly sunshine hours at The South Pole total 2,698.2 for the year.
  Fact 0.19: The mean daily sunshine hours at The South Pole are 13.1 in January, 17.6 in February, 6.3 in March, 0.0 in April, 0.0 in May, 0.0 in June, 0.0 in July, 0.0 in August, 1.1 in September, 12.6 in October, 18.6 in November and 19.9 in December.
  Fact 0.20: The mean daily sunshine hours at The South Pole are 7.4 for the year.
  Fact 0.21: Source 1 for the data is Pogoda.ru.net, for temperatures from 1981 to 2010 and extremes from 1957 to present.
  Fact 0.22: Source 2 for the data is Deutscher Wetterdienst, for precipitation from 1957 to 1988 and sun from 1978 to 1993.
  Fact 0.23: Source 2 also includes NOAA, for snowy days data from 1961 to 1988.

Retrieved items:
<item 1>
South Pole
The highest temperature ever recorded at the Amundsen -- Scott South Pole Station was − 12.3 ° C ( 9.9 ° F ) on Christmas Day , 2011 , and the lowest was − 82.8
° C ( − 117.0 ° F ) on June 23 , 1982 ( for comparison , the lowest temperature directly recorded anywhere on earth was − 89.2 ° C ( − 128.6 ° F ) at Vostok
Station on July 21 , 1983 , though − 93.2 ° C ( − 135.8 ° F ) was measured indirectly by satellite in East Antarctica between Dome A and Dome F in August 2010 )
. The South Pole has a desert climate , receiving very little precipitation . Air humidity is near zero . However , high winds can cause the blowing of snowfall
, and the accumulation of snow amounts to about 20 cm ( 8 in ) per year . The former dome seen in pictures of the Amundsen -- Scott station is partially buried
due to snow storms , and the entrance to the dome had to be regularly bulldozed to uncover it .
</item 1>

<item 2>
South Pole
0 0.0 1.1 12.6 18.6 19.9 7.4 Source # 1 : Pogoda.ru.net ( temperatures , 1981 -- 2010 , extremes 1957 -- present ) Source # 2 : Deutscher Wetterdienst (
precipitation 1957 -- 1988 and sun 1978 -- 1993 ) , NOAA ( snowy days data , 1961 -- 1988 ) Time ( edit ) In most places on Earth , local time is determined by
longitude , such that the time of day is more - or-less synchronised to the position of the sun in the sky ( for example , at midday the sun is roughly at its
highest ) . This line of reasoning fails at the South Pole , where the sun rises and sets only once per year , and all lines of longitude , and hence all time
zones , converge . There is no a priori reason for placing the South Pole in any particular time zone , but as a matter of practical convenience the Amundsen --
Scott South Pole Station keeps New Zealand Time ( UTC + 12 ) .
</item 2>

<item 3>
South Pole
1 ( − 42 ) − 58.9 ( − 74 ) − 71.1 ( − 96 ) − 75.0 ( − 103 ) − 78.3 ( − 108.9 ) − 82.8 ( − 117 ) − 80.6 ( − 113.1 ) − 79.3 ( − 110.7 ) − 79.4 ( − 110.9 ) − 72.0
( − 97.6 ) − 55.0 ( − 67 ) − 41.1 ( − 42 ) − 82.8 ( − 117 ) Average precipitation mm ( inches ) 0.3 ( 0.012 ) 0.6 ( 0.024 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) 0.2 (
0.008 ) 0.1 ( 0.004 ) Trace Trace 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.3 ( 0.012 ) 2.3 ( 0.
</item 3>

<item 4>
South Pole
. NIWA . Retrieved 2012 - 08 - 13 . Jump up ^ `` Antarctic Weather '' . Retrieved 2013 - 05 - 25 . Jump up ^ `` NASA - USGS Landsat 8 Satellite Pinpoints
Coldest Spots on Earth '' , NASA , December 9 , 2013 Jump up ^ Initial environmental evaluation -- development of blue - ice and compacted - snow runways ,
National Science Foundation Office of Polar Programs , April 9 , 1993 Jump up ^ `` Weather and Climate - The Climate of Amundsen -- Scott '' ( in Russian ) .
Weather and Climate ( Погода и климат ) . Retrieved 5 April 2017 . Jump up ^ `` Klimatafel von Amundsen - Scott / Südpol - Station ( USA ) / Antarktis '' ( PDF
) . Baseline climate means ( 1961 - 1990 ) from stations all over the world ( in German ) . Deutscher Wetterdienst . Retrieved 5 April 2017 .
</item 4>

<item 5>
South Pole
More recent buildings are raised on stilts so that the snow does not build up against the sides of them . ( hide ) Climate data for The South Pole Month Jan Feb
Mar Apr May Jun Jul Aug Sep Oct Nov Dec Year Record high ° C ( ° F ) − 14.4 ( 6.1 ) − 20.6 ( − 5.1 ) − 26.7 ( − 16.1 ) − 27.8 ( − 18 ) − 25.1 ( − 13.2 ) − 28.8
( − 19.8 ) − 33.9 ( − 29 ) − 32.8 ( − 27 ) − 29.3 ( − 20.7 ) − 25.1 ( − 13.2 ) − 18.9 ( − 2 ) − 12.3 ( 9.9 ) − 12.3 ( 9.9 ) Average high ° C ( ° F ) − 26.0 ( −
14.8 ) − 37.9 ( − 36.2 ) − 49.6 ( − 57.3 ) − 53.
</item 5>

<item 6>
South Pole
Deutscher Wetterdienst . Retrieved 5 April 2017 . Jump up ^ `` Amundsen -- Scott Climate Normals 1961 − 1990 '' . National Oceanic and Atmospheric
Administration . Retrieved 5 April 2017 . Jump up ^ Mark Sabbatini , `` Non-human life form seen at Pole '' , The Antarctic Sun , 5 January 2003 . Jump up ^ ``
Snow microbes found at South Pole '' , BBC News , 10 July 2000 External links ( edit ) Wikimedia Commons has media related to South Pole . Wikivoyage has a
travel guide for South Pole . NOAA South Pole Webcam 360 ° Panoramas of the South Pole Images of this location are available at the Degree Confluence Project
South Pole Photo Gallery Poles by the Australian Antarctic Division The Antarctic Sun -- Online news source for the U.S. Antarctic Program Big Dead Place UK
team makes polar trek history -- BBC News article on first expedition to Pole of Inaccessibility without mechanical assistance Listen to Ernest Shackleton
describing his 1908 South Pole Expedition , and read more about the recording on ( australianscreen online ) .
</item 6>

<item 7>
South Pole
2 ) − 49.6 ( − 57.3 ) − 53.0 ( − 63.4 ) − 53.6 ( − 64.5 ) − 54.5 ( − 66.1 ) − 55.2 ( − 67.4 ) − 54.9 ( − 66.8 ) − 54.4 ( − 65.9 ) − 48.4 ( − 55.1 ) − 36.2 ( −
33.2 ) − 26.3 ( − 15.3 ) − 45.8 ( − 50.4 ) Daily mean ° C ( ° F ) − 28.4 ( − 19.1 ) − 40.9 ( − 41.6 ) − 53.7 ( − 64.7 ) − 57.8 ( − 72 ) − 58.0 ( − 72.4 ) − 58.9
( − 74 ) − 59.8 ( − 75.6 ) − 59.7 ( − 75.5 ) − 59.1 ( − 74.4 ) − 51.
</item 7>

<item 8>
South Pole
However , it is always low in the sky , reaching a maximum of 23.5 ° in December . Much of the sunlight that does reach the surface is reflected by the white
snow . This lack of warmth from the sun , combined with the high altitude ( about 2,800 metres ( 9,200 ft ) ) , means that the South Pole has one of the coldest
climates on Earth ( though it is not quite the coldest ; that record goes to the region in the vicinity of the Vostok Station , also in Antarctica , which lies
at a higher elevation ) . Temperatures at the South Pole are much lower than at the North Pole , primarily because the South Pole is located at altitude in the
middle of a continental land mass , while the North Pole is at sea level in the middle of an ocean , which acts as a reservoir of heat . The South Pole is at an
altitude of 9,300 feet ( 2,800 m ) but feels like 11,000 feet ( 3,400 m ). . Centrifugal force from the spin of the planet pulls the atmosphere toward the
equator .
</item 8>

<item 9>
South Pole
Retrieved 2013 - 05 - 25 . Jump up ^ `` Crossing the Ice '' . Retrieved 2013 - 05 - 25 . Jump up ^ `` Wilson , nå er vi framme ! '' . Retrieved 2013 - 05 - 25 .
Jump up ^ Science question of the week , Goddard Space Flight Center . Jump up ^ `` The USAP Portal : Science and Support in Antarctica - Course Material '' .
www.usap.gov . Jump up ^ `` Why is the South Pole colder than the North Pole ? '' . Jump up ^ Matthew A. Lazzara ( 2011 - 12 - 28 ) . `` Preliminary Report :
Record Temperatures at South Pole ( and nearby AWS sites ... ) '' . Retrieved 2011 - 12 - 28 . Jump up ^ Your stay at Amundsen -- Scott South Pole Station
Archived 2006 - 02 - 02 at the Wayback Machine. , National Science Foundation Office of Polar Programs Jump up ^ `` How cold is the Antarctic ? '' . NIWA .
</item 9>

<item 10>
South Pole
The polar ice sheet is moving at a rate of roughly 10 metres per year in a direction between 37 ° and 40 ° west of grid north , down towards the Weddell Sea .
Therefore , the position of the station and other artificial features relative to the geographic pole gradually shift over time . The Geographic South Pole is
marked by a stake in the ice alongside a small sign ; these are repositioned each year in a ceremony on New Year 's Day to compensate for the movement of the
ice . The sign records the respective dates that Roald Amundsen and Robert F. Scott reached the Pole , followed by a short quotation from each man , and gives
the elevation as `` 9,301 FT . '' . A new marker stake is designed and fabricated each year by staff at the site . Ceremonial South Pole ( edit ) The Ceremonial
South Pole is an area set aside for photo opportunities at the South Pole Station . It is located some meters from the Geographic South Pole , and consists of a
metallic sphere on a short bamboo pole , surrounded by the flags of the original Antarctic Treaty signatory states .
</item 10>
=== END ITEM id=46e51e8e9350490f ===

=== ITEM id=a6e74949ff32ec00 ===
Gold sentences and facts:
Sentence 0: Arthur `` The President '' Flanders -- An inmate on death row , convicted of killing his father in an insurance - fraud scheme . His sentence is commuted to life imprisonment , but he is later murdered by an unknown inmate in the prison laundry . Arlen `` The Chief '' Bitterbuck -- A Washita Cherokee death - row inmate , convicted of killing a man in a drunken brawl over a pair of boots . His execution is the first of three mentioned in Paul 's story . Janice Edgecombe -- Paul Edgecombe 's wife . Dies in a bus accident on the way to her grandchild 's graduation in 1956 . Hal Moores -- The warden at Cold Mountain Penitentiary . Melinda Moores -- Warden Moores 's wife , who is dying of a brain tumor and is cured by John Coffey . She eventually dies of a heart attack ten years later . Curtis Anderson -- The assistant warden . Dean Stanton -- A guard on E Block who is strangled and nearly killed by William Wharton . A father of young children , he takes no part in the taking of John Coffey to Melinda Moores due to the risk of losing his job . He applies for relocation to C Block after John Coffey 's death , where he is murdered by an inmate 4 months later . Harry Terwilliger -- One of the main guards on E Block along with his friends Paul Edgecombe , Brutus Howell and Dean Stanton . He eventually dies of cancer , long after John Coffey 's execution . Bill Dodge -- A `` floater '' guard on E Block ( not permanently assigned there ) . Jack Van Hay -- A guard who is part of the execution team . He operates the switch room . Toot - Toot -- A trustee who stands in for the condemned during execution rehearsals and sells snacks to prisoners and guards . Burt Hammersmith -- A reporter who wrote on the Detterick twins ' murders and John Coffey 's trial . Despite believing himself to be an `` Enlightenment '' man , he displays prejudice in his stance on `` Negroes '' and tries to convince Paul of John Coffey 's guilt . Elaine Connelly -- A friend of Paul in the present - day nursing home where he tells his story . She is later revealed to be the grandmother of the Speaker of the Georgia House of Representatives , using this position to scare Brad Dolan from harassing Paul . Homer Cribus -- The Sheriff of Trapingus County , where the murders of the Detterick twins took place . Despite playing no part in apprehending John Coffey , he later shows up at the execution . An outspoken Baptist with strong racial prejudice and immensely overweight , he later succumbs to a heart attack while having sex with a 17 - year - old African American in his office . Rob McGee -- The Deputy Sheriff of Trapingus County , who led the search party which found John Coffey . While displaying strong doubt about Coffey 's guilt after being shown signs of innocence by Paul Edgecombe , he is still powerless to call for an appeal as he is subordinate to Sheriff Cribus , whom he allegedly hopes to succeed . Brad Dolan -- A malicious nursing home employee who harasses Paul Edgecombe . Paul strongly compares him to Percy Wetmore , and several times mistakes him for Percy , despite the latter dying in 1965 . Kathe and Cora Detterick -- The two young girls whom John Coffey was convicted of raping and murdering . Killed by William Wharton . Klaus and Marjorie Detterick -- The parents of Kathe and Cora .
  Fact 0.0: Arthur 'The President' Flanders is an inmate on death row.
  Fact 0.1: Arthur Flanders was convicted of killing his father in an insurance-fraud scheme.
  Fact 0.2: Arthur Flanders's sentence was commuted to life imprisonment.
  Fact 0.3: Arthur Flanders was later murdered by an unknown inmate.
  Fact 0.4: Arthur Flanders was murdered in the prison laundry.
  Fact 0.5: Arlen 'The Chief' Bitterbuck is a Washita Cherokee death-row inmate.
  Fact 0.6: Arlen Bitterbuck was convicted of killing a man in a drunken brawl over a pair of boots.
  Fact 0.7: Arlen Bitterbuck's execution is the first of three mentioned in Paul's story.
  Fact 0.8: Janice Edgecombe is Paul Edgecombe's wife.
  Fact 0.9: Janice Edgecombe dies in a bus accident on the way to her grandchild's graduation.
  Fact 0.10: Janice Edgecombe's death occurred in 1956.
  Fact 0.11: Hal Moores is the warden at Cold Mountain Penitentiary.
  Fact 0.12: Melinda Moores is Warden Moores's wife.
  Fact 0.13: Melinda Moores is dying of a brain tumor.
  Fact 0.14: John Coffey cures Melinda Moores of the brain tumor.
  Fact 0.15: Melinda Moores dies of a heart attack ten years later.
  Fact 0.16: Curtis Anderson is the assistant warden.
  Fact 0.17: Dean Stanton is a guard on E Block.
  Fact 0.18: William Wharton strangles Dean Stanton and nearly kills him.
  Fact 0.19: Dean Stanton is a father of young children.
  Fact 0.20: Dean Stanton takes no part in taking John Coffey to Melinda Moores due to the risk of losing his job.
  Fact 0.21: Dean Stanton applies for relocation to C Block after John Coffey's death.
  Fact 0.22: Dean Stanton is murdered by an inmate on C Block 4 months later.
  Fact 0.23: Harry Terwilliger is one of the main guards on E Block.
  Fact 0.24: Harry Terwilliger is friends with Paul Edgecombe, Brutus Howell, and Dean Stanton.
  Fact 0.25: Harry Terwilliger dies of cancer.
  Fact 0.26: Harry Terwilliger's death came long after John Coffey's execution.
  Fact 0.27: Bill Dodge is a 'floater' guard on E Block.
  Fact 0.28: Bill Dodge is not permanently assigned to E Block.
  Fact 0.29: Jack Van Hay is a guard who is part of the execution team.
  Fact 0.30: Jack Van Hay operates the switch room.
  Fact 0.31: Toot-Toot is a trustee.
  Fact 0.32: Toot-Toot stands in for the condemned during execution rehearsals.
  Fact 0.33: Toot-Toot sells snacks to prisoners and guards.
  Fact 0.34: Burt Hammersmith is a reporter.
  Fact 0.35: Burt Hammersmith wrote on the Detterick twins' murders.
  Fact 0.36: Burt Hammersmith wrote on John Coffey's trial.
  Fact 0.37: Burt Hammersmith believes himself to be an 'Enlightenment' man.
  Fact 0.38: Burt Hammersmith displays prejudice in his stance on 'Negroes'.
  Fact 0.39: Burt Hammersmith tries to convince Paul Edgecombe of John Coffey's guilt.
  Fact 0.40: Elaine Connelly is a friend of Paul Edgecombe in the present-day nursing home where he tells his story.
  Fact 0.41: Elaine Connelly is later revealed to be the grandmother of the Speaker of the Georgia House of Representatives.
  Fact 0.42: Elaine Connelly uses this position to scare Brad Dolan from harassing Paul.
  Fact 0.43: Homer Cribus is the Sheriff of Trapingus County.
  Fact 0.44: The Detterick twins' murders took place in Trapingus County.
  Fact 0.45: Homer Cribus played no part in apprehending John Coffey.
  Fact 0.46: Homer Cribus later shows up at the execution.
  Fact 0.47: Homer Cribus is an outspoken Baptist.
  Fact 0.48: Homer Cribus has strong racial prejudice.
  Fact 0.49: Homer Cribus is immensely overweight.
  Fact 0.50: Homer Cribus later succumbs to a heart attack while having sex with a 17-year-old African American in his office.
  Fact 0.51: Rob McGee is the Deputy Sheriff of Trapingus County.
  Fact 0.52: Rob McGee led the search party which found John Coffey.
  Fact 0.53: Rob McGee displays strong doubt about Coffey's guilt after being shown signs of innocence by Paul Edgecombe.
  Fact 0.54: Rob McGee is powerless to call for an appeal.
  Fact 0.55: Rob McGee is subordinate to Sheriff Cribus.
  Fact 0.56: Rob McGee allegedly hopes to succeed Sheriff Cribus.
  Fact 0.57: Brad Dolan is a malicious nursing home employee.
  Fact 0.58: Brad Dolan harasses Paul Edgecombe.
  Fact 0.59: Paul Edgecombe strongly compares Brad Dolan to Percy Wetmore.
  Fact 0.60: Paul Edgecombe several times mistakes Brad Dolan for Percy Wetmore.
  Fact 0.61: Percy Wetmore died in 1965.
  Fact 0.62: Kathe Detterick and Cora Detterick are two young girls.
  Fact 0.63: John Coffey was convicted of raping and murdering Kathe and Cora Detterick.
  Fact 0.64: William Wharton killed Kathe and Cora Detterick.
  Fact 0.65: Klaus Detterick and Marjorie Detterick are the parents of Kathe and Cora Detterick.
Sentence 1: Toot - Toot -- A trustee who stands in for the condemned during execution rehearsals and sells snacks to prisoners and guards .
  Fact 1.0: Toot-Toot is a trustee.
  Fact 1.1: Toot-Toot stands in for the condemned during execution rehearsals.
  Fact 1.2: Toot-Toot sells snacks to prisoners and guards.

Retrieved items:
<item 1>
The Green Mile (novel)
He is the first character to discover John Coffey 's amazing abilities , when the prisoner cures his urinary tract infection . It is also his idea to take
Coffey to try to cure Melinda , Warden Hal Moores ' wife , of her brain tumor . He transferred to Boy 's Correctional with Brutus Howell shortly after Coffey 's
execution . Brutus `` Brutal '' Howell -- He is second in command on the Green Mile and Paul 's best friend . He is a tall , imposing man but not violent at all
unless necessary . His nickname of `` Brutal '' is intended as irony . A former football tackle who had gone on to play at Louisiana State University ( LSU ) in
his youth , he eventually suffers a fatal heart attack at home , in his fifties , about twenty - five years after Coffey 's execution . John Coffey -- He is a
massive black man ( 6 ft 8 in tall ) , on death row for the alleged rape and murder of two young girls . He is very quiet and prefers to keep to himself , weeps
almost constantly , and is afraid of the dark .
</item 1>

<item 2>
The Green Mile (novel)
Brad Dolan -- A malicious nursing home employee who harasses Paul Edgecombe . Paul strongly compares him to Percy Wetmore , and several times mistakes him for
Percy , despite the latter dying in 1965 . Kathe and Cora Detterick -- The two young girls whom John Coffey was convicted of raping and murdering . Killed by
William Wharton . Klaus and Marjorie Detterick -- The parents of Kathe and Cora . Reception ( edit ) The Green Mile won the Bram Stoker Award for Best Novel in
1996 . In 1997 , The Green Mile was nominated as Best Novel for the British Fantasy Award and the Locus Award . In 2003 the book was listed on the BBC 's The
Big Read poll of the UK 's `` best - loved novel . '' Film adaptation ( edit ) Main article : The Green Mile ( film ) Frank Darabont adapted the novel into a
screenplay for a feature film of the same name . Released in 1999 , the film was directed by Darabont and starred Tom Hanks as Paul Edgecombe and Michael Clarke
Duncan as John Coffey .
</item 2>

<item 3>
The Green Mile (novel)
Even at the end , during his execution , he asks Paul Edgecombe not to put on the traditional black silk mask used to block the view of the prisoner 's face
because he fears the dark . Coffey is described as `` knowing his own name and not much else '' and lacks the capability to do so much as tie a simple knot .
However , he is convicted of luring the girls away from their home , disposing of the watchdog , carefully planning and using abilities he would otherwise not
be expected to have . He is the calmest and mildest prisoner the guards have ever seen , despite his hulking form . He turns out to be innocent of what he is
accused of , but chooses to die anyway . Percy Wetmore -- He is a young and sadistic guard . He is disliked by Paul and the other guards because of his ways ,
but they can not do anything about it because he is the nephew of the governor 's wife . He is very homophobic and attacks Eduard Delacroix for allegedly
touching him , although it was an accident caused by Del stumbling out of the prison truck .
</item 3>

<item 4>
The Green Mile (novel)
This year marks the arrival of John Coffey , a 6 ft 8 in powerfully built black man who has been convicted of raping and murdering two small white girls .
During his time on the Mile , John interacts with fellow prisoners Eduard `` Del '' Delacroix , a Cajun arsonist , rapist , and murderer , and William Wharton (
`` Billy the Kid '' to himself , `` Wild Bill '' to the guards ) , a wild - acting and dangerous multiple murderer who is determined to make as much trouble as
he can before he is executed . Other inhabitants include Arlen Bitterbuck , a Native American convicted of killing a man in a fight over a pair of boots ( also
the first character to die in the electric chair ) ; Arthur Flanders , a real estate executive who killed his father to perpetrate insurance fraud , and whose
sentence is eventually commuted to life imprisonment ( while serving his sentence , he is killed by another inmate in the laundry room ) ; and Mr. Jingles , a
mouse , to whom Del teaches various tricks . Paul and the other guards are irritated throughout the book by Percy Wetmore , a sadistic guard who enjoys
antagonizing the prisoners .
</item 4>

<item 5>
The Green Mile (novel)
Paul is unsure how to help John , but John tells him not to worry , as he is ready to die anyway , wanting to escape the cruelty of the world . John 's
execution is the last one in which Paul participates . He introduces Mr. Jingles to Elaine just before the mouse dies , having lived 64 years past these events
, and explains that those healed by John gained an unnaturally long lifespan . Elaine dies shortly after , never learning how Paul 's wife died in his arms
immediately after they suffered a bus accident , and that he then saw John Coffey 's ghost watching him from an overpass . Paul seems to be all alone , now 104
years old , and wondering how much longer he will live . Characters ( edit ) Paul Edgecombe -- The protagonist and narrator of the book and the death - row
supervisor at Cold Mountain Penitentiary . He is 40 years old when the main bulk of the story takes place , in 1932 . He is a caring man and takes excellent
care of the men on his block , avoiding conflict and keeping the peace whenever possible .
</item 5>

<item 6>
The Green Mile (novel)
He is later attacked by `` Wild Bill '' Wharton , consequently wets himself , and is teased by Delacroix for it . In retaliation , Percy deliberately sabotages
Delacroix 's execution . At the end of the story he is sent to the Briar Ridge mental institution , originally considered for a job but now as a patient , after
Coffey transferred Melinda 's disease to him which caused him to kill William Wharton . He eventually lives through a hospital fire and dies in 1965 . William
`` Wild Bill '' Wharton -- The antagonist of the story . He is on death row for various crimes . He does not like the nickname `` Wild Bill '' but prefers to be
called `` Billy the Kid '' , a name which he has tattooed on his forearm . When he first arrives he manages to convince the guards that he is in a drugged
stupor , only to attack and attempt to strangle to death Dean Stanton when they reach E Block . He continues to wreak havoc on the Mile and plays tricks such as
urinating on the guards , amongst other things . He is punished by being placed in solitary confinement , but never seems to learn his lesson .
</item 6>

<item 7>
The Green Mile (novel)
Curtis Anderson -- The assistant warden . Dean Stanton -- A guard on E Block who is strangled and nearly killed by William Wharton . A father of young children
, he takes no part in the taking of John Coffey to Melinda Moores due to the risk of losing his job . He applies for relocation to C Block after John Coffey 's
death , where he is murdered by an inmate 4 months later . Harry Terwilliger -- One of the main guards on E Block along with his friends Paul Edgecombe , Brutus
Howell and Dean Stanton . He eventually dies of cancer , long after John Coffey 's execution . Bill Dodge -- A `` floater '' guard on E Block ( not permanently
assigned there ) . Jack Van Hay -- A guard who is part of the execution team . He operates the switch room . Toot - Toot -- A trustee who stands in for the
condemned during execution rehearsals and sells snacks to prisoners and guards . Burt Hammersmith -- A reporter who wrote on the Detterick twins ' murders and
John Coffey 's trial .
</item 7>

<item 8>
The Green Mile (novel)
He is resurrected by John Coffey after being stomped on by Percy Wetmore . This gives him increased longevity and he finally dies 64 years later . Other
characters ( edit ) Arthur `` The President '' Flanders -- An inmate on death row , convicted of killing his father in an insurance - fraud scheme . His
sentence is commuted to life imprisonment , but he is later murdered by an unknown inmate in the prison laundry . Arlen `` The Chief '' Bitterbuck -- A Washita
Cherokee death - row inmate , convicted of killing a man in a drunken brawl over a pair of boots . His execution is the first of three mentioned in Paul 's
story . Janice Edgecombe -- Paul Edgecombe 's wife . Dies in a bus accident on the way to her grandchild 's graduation in 1956 . Hal Moores -- The warden at
Cold Mountain Penitentiary . Melinda Moores -- Warden Moores 's wife , who is dying of a brain tumor and is cured by John Coffey . She eventually dies of a
heart attack ten years later . Curtis Anderson -- The assistant warden .
</item 8>

<item 9>
The Green Mile (novel)
Despite believing himself to be an `` Enlightenment '' man , he displays prejudice in his stance on `` Negroes '' and tries to convince Paul of John Coffey 's
guilt . Elaine Connelly -- A friend of Paul in the present - day nursing home where he tells his story . She is later revealed to be the grandmother of the
Speaker of the Georgia House of Representatives , using this position to scare Brad Dolan from harassing Paul . Homer Cribus -- The Sheriff of Trapingus County
, where the murders of the Detterick twins took place . Despite playing no part in apprehending John Coffey , he later shows up at the execution . An outspoken
Baptist with strong racial prejudice and immensely overweight , he later succumbs to a heart attack while having sex with a 17 - year - old African American in
his office . Rob McGee -- The Deputy Sheriff of Trapingus County , who led the search party which found John Coffey . While displaying strong doubt about Coffey
's guilt after being shown signs of innocence by Paul Edgecombe , he is still powerless to call for an appeal as he is subordinate to Sheriff Cribus , whom he
allegedly hopes to succeed .
</item 9>

<item 10>
The Green Mile (novel)
He is punished by being placed in solitary confinement , but never seems to learn his lesson . As John Coffey is being smuggled to Hal Moores 's house , Wild
Bill grabs his arm and Coffey sees that he actually committed the murders Coffey was accused of . Therefore , Coffey gives Percy the `` sickness '' he took from
the warden 's wife , causing him to kill Wharton . Eduard `` Del '' Delacroix -- He is a Cajun prisoner with a fairly slow grasp of the English language . He is
incarcerated for multiple deaths due to a fire he started while trying to cover up his rape and murder of a young girl . While on the Mile , Del befriends a
mouse named Mr. Jingles , who becomes his best friend in his last days on death row . Percy , his enemy , sabotages his execution , causing Del to die in a slow
, gruesome death in the electric chair . Mr. Jingles ( nicknamed `` Steamboat Willie '' ) -- An unusually intelligent mouse who enjoys eating peppermint sweets
. He becomes a friend to Eduard Delacroix in the few days before the man is executed .
</item 10>
=== END ITEM id=a6e74949ff32ec00 ===

=== ITEM id=b26c49ecec636cca ===
Gold sentences and facts:
Sentence 0: Late in the 20th century , lithium became an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge - and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5 volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries , which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium - ion polymer battery , lithium iron phosphate battery , and the nanowire battery .
  Fact 0.0: Late in the 20th century, lithium became an important component of battery electrolytes.
  Fact 0.1: Late in the 20th century, lithium became an important component of battery electrodes.
  Fact 0.2: Lithium became an important battery component because of its high electrode potential.
  Fact 0.3: Lithium has a low atomic mass.
  Fact 0.4: Because of its low atomic mass, lithium has a high charge-to-weight ratio.
  Fact 0.5: Because of its low atomic mass, lithium has a high power-to-weight ratio.
  Fact 0.6: A typical lithium-ion battery can generate approximately 3 volts per cell.
  Fact 0.7: Lead-acid batteries generate 2.1 volts.
  Fact 0.8: Zinc-carbon cells generate 1.5 volts.
  Fact 0.9: Lithium-ion batteries are rechargeable.
  Fact 0.10: Lithium-ion batteries have a high energy density.
  Fact 0.11: Lithium-ion batteries should not be confused with lithium batteries.
  Fact 0.12: Lithium batteries are disposable (primary) batteries.
  Fact 0.13: Lithium batteries have lithium or its compounds as the anode.
  Fact 0.14: Other rechargeable batteries that use lithium include the lithium-ion polymer battery.
  Fact 0.15: Other rechargeable batteries that use lithium include the lithium iron phosphate battery.
  Fact 0.16: Other rechargeable batteries that use lithium include the nanowire battery.
Sentence 1: Lithium and its compounds have several industrial applications , including heat - resistant glass and ceramics , lithium grease lubricants , flux additives for iron , steel and aluminium production , lithium batteries , and lithium - ion batteries . These uses consume more than three quarters of lithium production .
  Fact 1.0: Lithium and its compounds have several industrial applications.
  Fact 1.1: Lithium and its compounds are used in heat-resistant glass and ceramics.
  Fact 1.2: Lithium and its compounds are used in lithium grease lubricants.
  Fact 1.3: Lithium and its compounds are used as flux additives for iron, steel and aluminium production.
  Fact 1.4: Lithium and its compounds are used in lithium batteries.
  Fact 1.5: Lithium and its compounds are used in lithium-ion batteries.
  Fact 1.6: These uses consume more than three quarters of lithium production.

Retrieved items:
<item 1>
Lithium
Lithium - wikipedia Lithium Jump to : navigation , search This article is about the chemical element . For the use of lithium as a medication , see Lithium (
medication ) . For other uses , see Lithium ( disambiguation ) . It has been suggested that Lithium as an investment be merged into this article . ( Discuss )
Proposed since June 2017 . Lithium , Li Lithium floating in oil General properties Pronunciation / ˈlɪθiəm / ​ ( LI - thee - əm ) Appearance silvery - white
Standard atomic weight ( A ) ( 7000693800000000000 ♠ 6.938 , 7000699700000000000 ♠ 6.997 ) conventional : 7000694000000000000 ♠ 6.94 Lithium in the periodic
table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium
Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium
</item 1>

<item 2>
Lithium
When cut open , it exhibits a metallic luster , but moist air corrodes it quickly to a dull silvery gray , then black tarnish . It never occurs freely in nature
, but only in ( usually ionic ) compounds , such as pegmatitic minerals which were once the main source of lithium . Due to its solubility as an ion , it is
present in ocean water and is commonly obtained from brines . Lithium metal is isolated electrolytically from a mixture of lithium chloride and potassium
chloride . The nucleus of the lithium atom verges on instability , since the two stable lithium isotopes found in nature have among the lowest binding energies
per nucleon of all stable nuclides . Because of its relative nuclear instability , lithium is less common in the solar system than 25 of the first 32 chemical
elements even though its nuclei are very light : it is an exception to the trend that heavier nuclei are less common . For related reasons , lithium has
important uses in nuclear physics . The transmutation of lithium atoms to helium in 1932 was the first fully man - made nuclear reaction , and lithium deuteride
serves as a fusion fuel in staged thermonuclear weapons .
</item 2>

<item 3>
Lithium
Another significant mineral of lithium is lepidolite . A newer source for lithium is hectorite clay , the only active development of which is through the
Western Lithium Corporation in the United States . At 20 mg lithium per kg of Earth 's crust , lithium is the 25th most abundant element . According to the
Handbook of Lithium and Natural Calcium , `` Lithium is a comparatively rare element , although it is found in many rocks and some brines , but always in very
low concentrations . There are a fairly large number of both lithium mineral and brine deposits but only comparatively few of them are of actual or potential
commercial value . Many are very small , others are too low in grade . '' The US Geological Survey estimates that in 2010 , Chile had the largest reserves by
far ( 7.5 million tonnes ) and the highest annual production ( 8,800 tonnes ) . One of the largest reserve bases of lithium is in the Salar de Uyuni area of
Bolivia , which has 5.4 million tonnes . Other major suppliers include Australia , Argentina and China .
</item 3>

<item 4>
Lithium
Lithium hydroxide is a strong base and , when heated with a fat , produces a soap made of lithium stearate . Lithium soap has the ability to thicken oils , and
it is used to manufacture all - purpose , high - temperature lubricating greases . Metallurgy ( edit ) Lithium ( e.g. as lithium carbonate ) is used as an
additive to continuous casting mould flux slags where it increases fluidity , a use which accounts for 5 % of global lithium use ( 2011 ) . Lithium compounds
are also used as additives ( fluxes ) to foundry sand for iron casting to reduce veining . Lithium ( as lithium fluoride ) is used as an additive to aluminium
smelters ( Hall -- Héroult process ) , reducing melting temperature and increasing electrical resistance , a use which accounts for 3 % of production ( 2011 ) .
When used as a flux for welding or soldering , metallic lithium promotes the fusing of metals during the process and eliminates the forming of oxides by
absorbing impurities .
</item 4>

<item 5>
Lithium
Worldwide , this is one of the largest use for lithium compounds . Glazes containing lithium oxides are used for ovenware . Lithium carbonate ( Li CO ) is
generally used in this application because it converts to the oxide upon heating . Electrical and electronics ( edit ) Late in the 20th century , lithium became
an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge
- and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5
volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries
, which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium -
ion polymer battery , lithium iron phosphate battery , and the nanowire battery . Lubricating greases ( edit ) Main article : Lithium grease The third most
common use of lithium is in greases .
</item 5>

<item 6>
Lithium
2 10 cm / mol ( 298 K ) Young 's modulus 4.9 GPa Shear modulus 4.2 GPa Bulk modulus 11 GPa Mohs hardness 0.6 Brinell hardness 5 MPa CAS Number 7439 - 93 - 2
History Discovery Johan August Arfwedson ( 1817 ) First isolation William Thomas Brande ( 1821 ) Main isotopes of lithium Isotope Abundance Half - life ( t )
Decay mode Product Li 5 % stable Li 95 % stable Li content may be as low as 3.75 % in natural samples . Li would therefore have a content of up to 96.25 % .
view talk references in Wikidata Lithium ( from Greek : λίθος , translit . lithos , lit . ' stone ' ) is a chemical element with symbol Li and atomic number 3 .
It is a soft , silvery - white alkali metal . Under standard conditions , it is the lightest metal and the lightest solid element . Like all alkali metals ,
lithium is highly reactive and flammable , and is stored in mineral oil .
</item 6>

<item 7>
Lithium
The stockpiled lithium was depleted in lithium - 6 by 75 % , which was enough to affect the measured atomic weight of lithium in many standardized chemicals ,
and even the atomic weight of lithium in some `` natural sources '' of lithium ion which had been `` contaminated '' by lithium salts discharged from isotope
separation facilities , which had found its way into ground water . Lithium was used to decrease the melting temperature of glass and to improve the melting
behavior of aluminium oxide when using the Hall - Héroult process . These two uses dominated the market until the middle of the 1990s . After the end of the
nuclear arms race , the demand for lithium decreased and the sale of Department of Energy stockpiles on the open market further reduced prices . In the mid
1990s , several companies started to extract lithium from brine which proved to be a less expensive than underground or open - pit mining . Most of the mines
closed or shifted their focus to other materials because only the ore from zoned pegmatites could be mined for a competitive price . For example , the US mines
near Kings Mountain , North Carolina closed before the beginning of the 21st century .
</item 7>

<item 8>
Lithium
Extraction ( edit ) Lithium salts are extracted from water in mineral springs , brine pools , and brine deposits . Lithium is present in seawater , but
commercially viable methods of extraction have yet to be developed . Another potential source of lithium is the leachates of geothermal wells , which are
carried to the surface . Recovery of lithium has been demonstrated in the field ; the lithium is separated by simple filtration . The process and environmental
costs are primarily those of the already - operating well ; net environmental impacts may thus be positive . Uses ( edit ) Estimates of global lithium uses in
2015 Ceramics and glass ( 32 % ) Batteries ( 35 % ) Lubricating greases ( 9 % ) Continuous casting ( 5 % ) Air treatment ( 5 % ) Polymers ( 4 % ) Primary
aluminum production ( 1 % ) Pharmaceuticals ( < 1 % ) Other ( 9 % ) Ceramics and glass ( edit ) Lithium oxide is widely used as a flux for processing silica ,
reducing the melting point and viscosity of the material and leading to glazes with improved physical properties including low coefficients of thermal expansion
.
</item 8>

<item 9>
Lithium
000,000 Portugal 200 60,000 N / A Russia - - 1,000,000 Serbia - - 1,000,000 United States 38,000 6,900,000 Zimbabwe 900 23,000 100,000 + World total 32,500
14,000,000 46,900,000 See also : Lithium minerals Although lithium is widely distributed on Earth , it does not naturally occur in elemental form due to its
high reactivity . The total lithium content of seawater is very large and is estimated as 230 billion tonnes , where the element exists at a relatively constant
concentration of 0.14 to 0.25 parts per million ( ppm ) , or 25 micromolar ; higher concentrations approaching 7 ppm are found near hydrothermal vents .
Estimates for the Earth 's crustal content range from 20 to 70 ppm by weight . In keeping with its name , lithium forms a minor part of igneous rocks , with the
largest concentrations in granites . Granitic pegmatites also provide the greatest abundance of lithium - containing minerals , with spodumene and petalite
being the most commercially viable sources .
</item 9>

<item 10>
Lithium
We are not aware of any discoveries of lithium , ' he said . '' Lithia ( `` lithium brine '' ) is associated with tin mining areas in Cornwall , England and an
evaluation project from 400 - metre deep test boreholes is under consideration . If successful the hot brines will also provide geothermal energy to power the
lithium extraction and refining process . Biological ( edit ) Lithium is found in trace amount in numerous plants , plankton , and invertebrates , at
concentrations of 69 to 5,760 parts per billion ( ppb ) . In vertebrates the concentration is slightly lower , and nearly all vertebrate tissue and body fluids
contain lithium ranging from 21 to 763 ppb . Marine organisms tend to bioaccumulate lithium more than terrestrial organisms . Whether lithium has a
physiological role in any of these organisms is unknown . History ( edit ) Johan August Arfwedson is credited with the discovery of lithium in 1817 Petalite (
LiAlSi O ) was discovered in 1800 by the Brazilian chemist and statesman José Bonifácio de Andrada e Silva in a mine on the island of Utö , Sweden .
</item 10>
=== END ITEM id=b26c49ecec636cca ===

=== ITEM id=f0575942d1f1b897 ===
Gold sentences and facts:
Sentence 0: A subcutaneous injection is administered as a bolus into the subcutis , the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut is the preferred abbreviation for patient safety .
  Fact 0.0: A subcutaneous injection is administered as a bolus into the subcutis.
  Fact 0.1: The subcutis is the layer of skin directly below the dermis and epidermis.
  Fact 0.2: The dermis and epidermis are collectively referred to as the cutis.
  Fact 0.3: Subcutaneous injections are highly effective in administering vaccines.
  Fact 0.4: Subcutaneous injections are highly effective in administering medications such as insulin, morphine, diacetylmorphine and goserelin.
  Fact 0.5: Subcutaneous injection of recreational drugs, as opposed to intravenous injection, is referred to as 'skin popping'.
  Fact 0.6: Subcutaneous administration may be abbreviated as SC, SQ, sub-cu, sub-Q, SubQ, or subcut.
  Fact 0.7: Subcut is the preferred abbreviation for patient safety.

Retrieved items:
<item 1>
Subcutaneous injection
Subcutaneous injection - wikipedia Subcutaneous injection Jump to : navigation , search `` Subcut '' redirects here . For the song by Grand Mixer DXT featuring
Bill Laswell , see Aftermathematics . Play media 3D animation of a subcutaneous injection A subcutaneous injection is administered as a bolus into the subcutis
, the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in
administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of
recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut
is the preferred abbreviation for patient safety . Subcutaneous tissue has few blood vessels and so drugs injected here are for slow , sustained rates of
absorption . It is slower than intramuscular injections but still faster than intradermal injections .
</item 1>

<item 2>
Gallbladder
This sits on a thin layer of connective tissue , the lamina propria . The mucosa is curved and collected into tiny outpouchings called rugae . A muscular layer
sits beneath the mucosa . This is formed by smooth muscle , with fibres that lie in longitudinal , oblique and transverse directions , and are not arranged in
separate layers . The muscle fibres here contract to expel bile from the gallbladder . A distinctive feature of the gallbladder is the presence of Rokitansky --
Aschoff sinuses , deep outpouchings of the mucosa that can extend through the muscular layer . The muscular layer is surrounded by a layer of connective and fat
tissue . The outer layer of the fundus of gallbladder , and the surfaces not in contact with the liver , are covered by a thick serosa , which is exposed to the
peritoneum . The serosa contains blood vessels and lymphatics . The surfaces in contact with the liver are covered in connective tissue . Variation ( edit ) The
gallbladder varies in size , shape , and position between different people .
</item 2>

<item 3>
Subcutaneous injection
25.2007. s112 . ( hide ) Routes of administration , dosage forms Oral Digestive tract ( enteral ) Solids Pill Tablet Capsule Pastille Time release technology
Osmotic delivery system ( OROS ) Liquids Decoction Elixir Electuary Emulsion Extended - release syrup Effervescent powder or tablet Herbal tea Hydrogel
Molecular encapsulation Powder Softgel Solution Suspension Syrup Syrup Concentrate for dilution and / or addition of carbonated water Tincture Buccal (
sublabial ) , sublingual Solids Orally disintegrating tablet ( ODT ) Film Lollipop Sublingual drops Lozenges Effervescent buccal tablet Chewing gum Liquids
Mouthwash Toothpaste Ointment Oral spray Respiratory tract Solids Smoking device Dry - powder inhaler ( DPI ) 0 0 Liquids Anaesthetic vaporizer Vaporizer
Nebulizer Metered - dose inhaler ( MDI ) Gas Oxygen mask and Nasal cannula Oxygen concentrator Anaesthetic machine Relative analgesia machine Ophthalmic ,
otologic ,
</item 3>

<item 4>
Subcutaneous injection
It is slower than intramuscular injections but still faster than intradermal injections . Contents ( hide ) 1 Injection sites 2 Equipment 3 Procedure 4
Subcutaneous insulin injections in diabetes mellitus 5 See also 6 References Injection sites ( edit ) Subcutaneous injection sites Sites include , The outer
area of the upper arm . The abdomen , from the rib margin to the iliac crest and avoiding a 2 - inch circle around the navel . This has the fastest rate of
absorption among the sites . The front of the thigh , midway to the outer side , 4 inches below the top of the thigh to 4 inches above the knee . This has a
slower rate of absorption than the upper arm . The upper back . The upper area of the buttock , just behind the hip bone . This has the slowest rate of
absorption among the sites . Equipment ( edit ) A 25 to 31 gauge thick , 3 / 8 '' to 1 '' long needle can be used . The size is determined by the amount of
subcutaneous tissue present , which is based on patient build .
</item 4>

<item 5>
Subcutaneous injection
The 3 / 8 '' and 5 / 8 '' needles are most commonly used . Usually , no more than 1 mL of solution is given , compared to intradermal injections , where no more
than 0.5 mL is usually given . Procedure ( edit ) Subcutaneous injections are inserted at 45 to 90 degree angles , depending on amount of subcutaneous tissue
present and length of needle - a shorter , 3 / 8 '' needle is usually inserted 90 degrees and a 5 / 8 '' needle is usually inserted at 45 degrees . Medication
is administered slowly , about 10 seconds / milliliter . Subcutaneous insulin injections in diabetes mellitus ( edit ) An insulin pump with a subcutaneous
injection site . A person with insulin - dependent ( either type 1 or type 2 ) diabetes mellitus typically injects insulin subcutaneously . The injection should
be given under the skin , into the fat layer .
</item 5>

<item 6>
Subcutaneous injection
otologic , nasal Nasal spray Ear drops Eye drops Ointment Hydrogel Nanosphere suspension Insufflation Mucoadhesive microdisc ( microsphere tablet ) Urogenital
Ointment Pessary ( vaginal suppository ) Vaginal ring Vaginal douche Intrauterine device ( IUD ) Extra-amniotic infusion Intravesical infusion Rectal ( enteral
) Ointment Suppository Enema Solution Hydrogel Murphy drip Nutrient enema Dermal Ointment Topical cream Topical gel Liniment Paste Film DMSO drug solution
Electrophoretic dermal delivery system Hydrogel Liposomes Transfersome vesicles Cream Lotion Lip balm Medicated shampoo Dermal patch Transdermal patch Contact (
rubbed into break in the skin ) Transdermal spray Jet injector Injection , infusion ( into tissue / blood ) Skin Intradermal Subcutaneous Transdermal implant
Organs Intracavernous Intravitreal Intra-articular injection Transscleral Central nervous system Intracerebral Intrathecal Epidural Circulatory ,
</item 6>

<item 7>
Serratus anterior muscle
The muscle is inserted along the medial border of the scapula between the superior and inferior angles along with being inserted along the thoracic vertebrae .
The muscle is divided into three named parts depending on their points of insertions : the serratus anterior superior is inserted near the superior angle the
serratus anterior intermediate is inserted along the medial border the serratus anterior inferior is inserted near the inferior angle . Relations ( edit ) The
serratus anterior lies deep to the subscapularis , from which it is separated by the subscapularis ( supraserratus ) bursa . It is separated from the rib by the
scapulothoracic ( infraserratus ) bursa . Nerve supply ( edit ) The serratus anterior is innervated by the long thoracic nerve ( Nerve of Bell ) , a branch of
the brachial plexus . The long thoracic nerve travels inferiorly on the surface of the serratus . The nerve is especially vulnerable during certain types of
surgery ( for example , during lymph node clearance from the axilla ( e.g.
</item 7>

<item 8>
Subcutaneous injection
The injection should be given under the skin , into the fat layer . See also ( edit ) Intramuscular injection Intravenous injection Intradermal injection
References ( edit ) Jump up ^ `` subcutaneous injection '' at Dorland 's Medical Dictionary Jump up ^ `` ISMP 's List of Error - Prone Abbreviations , Symbols ,
and Dose Designations '' ( PDF ) . www.ismp.org. 2013 . Retrieved 13 May 2013 . ^ Jump up to : Taylor , C.R. , Lillis , C. , LeMone , P. , Lynn , P. ( 2011 )
Fundamentals of nursing : The art and science of nursing care . Philadelphia : Lippincott Williams & Wilkins , page 749 . Jump up ^ `` Insulin Administration ''
. Diabetes Care . 25 ( suppl 1 ) : s112 -- s115 . January 2002 . doi : 10.2337 / diacare. 25.2007. s112 . ( hide ) Routes of administration ,
</item 8>

<item 9>
Subcutaneous injection
musculoskeletal Intravenous Intracardiac Intramuscular Intraosseous Intraperitoneal Nanocell injection Patient - Controlled Analgesia pump PIC line WikiProject
Pharmacy portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Subcutaneous_injection&oldid=827949994 '' Categories : Dosage forms Routes of
administration Injection ( medicine ) Talk Contents About Wikipedia Dansk Deutsch Español فارسی Français Italiano Nederlands Norsk Polski Português Slovenščina
Türkçe Українська Tiếng Việt 6 more Edit links This page was last edited on 27 February 2018 , at 18 : 16 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 10>
=== END ITEM id=f0575942d1f1b897 ===
