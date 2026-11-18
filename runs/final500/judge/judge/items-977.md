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

=== ITEM id=0cf0b5c5023567e5 ===
Gold sentences and facts:
Sentence 0: Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
  Fact 0.0: Uncle Tom's Cabin was serialized in The National Era for ten months.
  Fact 0.1: The serialization of Uncle Tom's Cabin in The National Era began with the June 5, 1851 issue.
  Fact 0.2: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852.
  Fact 0.3: The book edition of Uncle Tom's Cabin was published by John P. Jewett and Company.
Sentence 1: Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical , starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story 's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's Cabin in book form , she eventually consented to the request .
  Fact 1.0: Uncle Tom's Cabin first appeared as a 40-week serial in The National Era.
  Fact 1.1: The National Era was an abolitionist periodical.
  Fact 1.2: The serial started with the June 5, 1851 issue.
  Fact 1.3: Uncle Tom's Cabin was originally intended as a shorter narrative that would run for only a few weeks.
  Fact 1.4: Harriet Beecher Stowe expanded the story significantly.
  Fact 1.5: The expanded story was instantly popular.
  Fact 1.6: Several protests were sent to the National Era office when Stowe missed an issue.
  Fact 1.7: The publisher John P. Jewett contacted Stowe about turning the serial into a book because of the story's popularity.
  Fact 1.8: Stowe questioned whether anyone would read Uncle Tom's Cabin in book form.
  Fact 1.9: Stowe eventually consented to the request to publish the story as a book.
Sentence 2: Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or , Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company . Followed by A Key to Uncle Tom 's Cabin ( 1853 )
  Fact 2.0: The infobox author of Uncle Tom's Cabin is Harriet Beecher Stowe.
  Fact 2.1: The original title of Uncle Tom's Cabin is Uncle Tom's Cabin; or, Life Among the Lowly.
  Fact 2.2: The illustrator of the 1st edition of Uncle Tom's Cabin is Hammatt Billings.
  Fact 2.3: Uncle Tom's Cabin is from the United States.
  Fact 2.4: Uncle Tom's Cabin is written in English.
  Fact 2.5: Uncle Tom's Cabin is a novel.
  Fact 2.6: Uncle Tom's Cabin was serialized in The National Era for ten months, beginning with the June 5, 1851 issue.
  Fact 2.7: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852, by John P. Jewett and Company.
  Fact 2.8: Uncle Tom's Cabin is followed by A Key to Uncle Tom's Cabin (1853).
  Fact 2.9: The infobox includes an image of the Boston edition of Uncle Tom's Cabin.

Retrieved items:
<item 1>
Uncle Tom's Cabin
Uncle Tom 's Cabin - wikipedia Uncle Tom 's Cabin Jump to : navigation , search This article is about the mid-19th century novel . For other uses , see Uncle
Tom 's Cabin ( disambiguation ) . Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or ,
Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era
for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
</item 1>

<item 2>
Uncle Tom's Cabin
Stowe mentioned a number of the inspirations and sources for her novel in A Key to Uncle Tom 's Cabin ( 1853 ) . This non-fiction book was intended to verify
Stowe 's claims about slavery . However , later research indicated that Stowe did not read many of the book 's cited works until after she had published her
novel . Publication The National Era , June 5 , 1851 Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical ,
starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story
significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story
's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's
Cabin in book form , she eventually consented to the request .
</item 2>

<item 3>
Uncle Tom's Cabin
The book was translated into all major languages , and in the United States it became the second best - selling book after the Bible . A number of the early
editions carried an introduction by Rev James Sherman , a Congregational minister in London noted for his abolitionist views . Uncle Tom 's Cabin sold equally
well in Britain , with the first London edition appearing in May 1852 and selling 200,000 copies . In a few years over 1.5 million copies of the book were in
circulation in Britain , although most of these were infringing copies ( a similar situation occurred in the United States ) . Plot Eliza escapes with her son ,
Tom sold `` down the River '' Full - page illustration by Hammatt Billings for Uncle Tom 's Cabin depicts Eliza telling Uncle Tom that he has been sold and she
is running away to save her child ( first edition : Boston : John P. Jewett and Company , 1852 ) . The book opens with a Kentucky farmer named Arthur Shelby
facing the loss of his farm because of debts .
</item 3>

<item 4>
Uncle Tom's Cabin
4 ( January 1960 ) , pp. 421 -- 433 . Jump up ^ The Classic Text : Harriett Beecher Stowe . University of Wisconsin - Milwaukee Library . Special collection
page on traditions and interpretations of Uncle Tom 's Cabin . Retrieved May 15 , 2007 . Archived May 16 , 2008 , at the Wayback Machine . ^ Jump up to : Uncle
Tom 's Cabin , introduction by Amanda Claybaugh , Barnes and Noble Classics , New York , 2003 , p. xvii . Jump up ^ A.M. Arnett , `` Review of James Baird
Weaver by Fred Emory Haynes '' , Political Science Quarterly , Vol. 35 , No. 1 ( March 1920 ) , pp. 154 -- 157 ; and profile of James Baird Weaver Archived
January 2 , 2007 , at the Wayback Machine. , accessed February 17 , 2007 . Jump up ^ Nassau Senior , quoted in Ephraim Douglass Adams , Great Britain and the
American Civil War ( 1958 ) , p. 33 .
</item 4>

<item 5>
Uncle Tom's Cabin
Convinced the book would be popular , Jewett made the unusual decision ( for the time ) to have six full - page illustrations by Hammatt Billings engraved for
the first printing . Published in book form on March 20 , 1852 , the novel sold 3,000 copies on that day alone , and soon sold out its complete print run . A
number of other editions were soon printed ( including a deluxe edition in 1853 , featuring 117 illustrations by Billings ) . In the first year of publication ,
300,000 copies of Uncle Tom 's Cabin were sold . At that point , however , `` demand came to an unexpected halt ... No more copies were produced for many years
, and if , as is claimed , Abraham Lincoln greeted Stowe in 1862 as ' the little woman who wrote the book that made this great war , ' the work had effectively
been out of print for many years . '' Jewett went out of business , and it was not until Ticknor and Fields put the work back in print in November 1862 that
demand began again to increase .
</item 5>

<item 6>
Uncle Tom's Cabin
Uncle Tom 's Cabin : or Life among the lowly ; frontispiece by John Gilbert ; ornamental title - page by Phiz ; and 130 engravings on wood by Matthew Urlwin
Sears , 1853 ( a searchable facsimile at the University of Georgia Libraries ; DjVu & layered PDF format ) Pictures and stories from Uncle Tom 's Cabin ; `` The
purpose of the editor of this little work , has been to adapt it for the juvenile family circle . The verses have accordingly been written by the authoress for
the capacity of the youngest readers ... '' 1853 ( a searchable facsimile at the University of Georgia Libraries ; DjVu & layered PDF format ) Uncle Tom 's
Cabin at Project Gutenberg Uncle Tom 's Cabin , available at Internet Archive . Scanned , illustrated original editions .
</item 6>

<item 7>
Uncle Tom's Cabin
42 . ^ Jump up to : Thomas F. Gossett , `` Review of The Building of Uncle Tom 's Cabin by E. Bruce Kirkham '' , American Literature , Vol. 50 , No. 1 ( March
1978 ) , pp. 123 -- 124 . Jump up ^ Charles Nichols , `` The Origins of Uncle Tom 's Cabin '' , The Phylon Quarterly , Vol. 19 , No. 3 ( 3rd Qtr , 1958 ) , p.
328 . ^ Jump up to : Jane Tompkins , `` Sentimental Power : Uncle Tom 's Cabin and the Politics of Literary History '' Archived December 16 , 2007 , at the
Wayback Machine. , from In Sensational Designs : The Cultural Work of American Fiction , 1790 -- 1860 . New York : Oxford University Press , 1985 , pp. 122 --
146 . Jump up ^ Posner , R. , Public Intellectuals : A Study of Decline , Harvard University Press ( 2002 ) , p. 239 . ISBN 0 - 674 - 00633 - X .
</item 7>

<item 8>
Uncle Tom's Cabin
ISSN 1551 - 5486 . ^ Jump up to : Henry Louis Gates , Kwame Anthony Appiah , Africana : Arts and Letters : An A-to - Z Reference of Writers , Musicians , and
Artists of the African American Experience , Running Press , 2005 , p. 544 . ^ Jump up to : Susan Logue , `` Historic Uncle Tom 's Cabin Saved '' , VOA News ,
January 12 , 2006 . Retrieved December 24 , 2011 . Jump up ^ Harriet Beecher Stowe , A Key to Uncle Tom 's Cabin 1853 , p. 42 , in which Stowe states : `` A
last instance parallel with that of Uncle Tom is to be found in the published memoirs of the venerable Josiah Henson ) ... '' This also is cited in Debra J.
Rosenthal , A Routledge Literary Sourcebook on Harriet Beecher Stowe 's Uncle Tom 's Cabin , Routledge , 2003 , pp. 25 -- 26 . Jump up ^ Annys Shin , `` After
buying historic home , Md.
</item 8>

<item 9>
Uncle Tom's Cabin
122 . Jump up ^ Gibson , Ian , The English Vice : Beating , Sex and Shame in Victorian England and After ( 1978 ) The passage which seems most likely to have
interested them is the start of chapter 29 . Jump up ^ `` La Cabaña del Tío Tomás '' ( PDF ) . Gaceta del Gobierno del Salvador en la América Central ( in
Spanish ) . Tomo 4 Número 17 . July 22 , 1853 . p. 4 . Archived from the original ( PDF ) on April 21 , 2018 . Retrieved April 21 , 2018 . Jump up ^ Tompkins ,
Jane . Sensational Designs : The Cultural Work of American Fiction , 1790 -- 1860 . New York : Oxford University Press , 1986 . See chapter five , ``
Sentimental Power : Uncle Tom 's Cabin and the Politics of Literary History . '' Jump up ^ Cindy Weinstein , The Cambridge Companion to Harriet Beecher Stowe ,
Cambridge University Press , 2004 , p. 13 .
</item 9>

<item 10>
Uncle Tom's Cabin
Parfait , Claire . The Publishing History of Uncle 's Tom 's Cabin , 1852 -- 2002 , Aldershot : Ashgate , 2007 . Reynolds , David S. Mightier Than the Sword :
Uncle Tom 's Cabin and the Battle for America , Norton , 2011 . 351 pp . Rosenthal , Debra J. Routledge Literary Sourcebook on Harriet Beecher Stowe 's Uncle
Tom 's Cabin , Routledge , 2003 . Sundquist , Eric J. , ed . New Essays on Uncle Tom 's Cabin , Cambridge University Press , 1986 . Tompkins , Jane . In
Sensational Designs : The Cultural Work of American Fiction , 1790 -- 1860 . New York : Oxford University Press , 1985 . Weinstein , Cindy . The Cambridge
Companion to Harriet Beecher Stowe , Cambridge University Press , 2004 . Williams , Linda . Playing the Race Card : Melodramas of Black and White from Uncle Tom
to O.J. Simpson , Princeton University Press , 2001 .
</item 10>
=== END ITEM id=0cf0b5c5023567e5 ===

=== ITEM id=01db2bc0fe7b2832 ===
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
This penalty of 20 % or 40 % of the increase in tax is due in the case of substantial understatement of tax , substantial valuation misstatements , transfer
pricing adjustments , or negligence or disregard of rules or regulations . For example , a valuation overstatement can result in a 30 % penalty on the amount of
tax owed . Special rules apply for each of these types of errors under which the penalty may be waived . Penalties in connection with information returns ( edit
) Certain types of returns , called `` information returns , '' do not require payment of tax . These include forms filed by employers to report wages ( Form W
- 2 ) and businesses to report certain payments ( Form 1099 series instructions ) . The penalty for failures related to these forms is a dollar amount per form
not timely filed , and the amount of penalty increases with the degree of lateness . The current maximum penalty for these forms is $50 . Many of the forms must
be filed electronically , and filing on paper is considered non-filing . Late filing of returns of partnership income ( Form 1065 ) can result in penalties of
$195 per month per partner , up to a maximum of 12 months .
</item 1>

<item 2>
IRS penalties
Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines . Contents ( hide ) 1 Underestimate and late payment
penalties 2 Penalties for failure to timely file returns or timely pay tax 3 Accuracy related penalties 4 Penalties in connection with information returns 5 100
% penalty on unpaid withholding taxes 6 Penalties for failure to provide foreign information 7 Excise taxes as penalties 8 Tax fraud penalties 9 Tax adviser
penalties 10 Judicial appeal of penalties 11 Further reading 12 References Underestimate and late payment penalties ( edit ) Taxpayers are required to have
withholding of tax or make quarterly estimated tax payments before the end of the tax year . Because accurate estimation requires accurate prediction of the
future , taxpayers sometimes underestimate the amount due . The penalty for paying too little estimated tax or having too little tax withheld is computed with
interest on the amount that should have been , but was not , paid . Where a taxpayer has filed an income or excise tax return that shows a balance due but does
not pay that balance by the due date of the return ( without extensions ) , a different charge applies .
</item 2>

<item 3>
IRS penalties
IRS penalties - wikipedia IRS penalties Jump to : navigation , search This article is part of a series on Taxation in the United States of America Federal
taxation ( show ) Alternative minimum tax Capital gains tax Corporate tax Estate tax Excise tax Gift tax Income tax Payroll tax Internal Revenue Service ( IRS )
Internal Revenue Code ( IRC ) IRS tax forms Revenue by state History Constitutional authority Taxpayer standing Court Protest Evasion State and local taxation (
show ) State income tax Property tax Sales tax Use tax Land value tax State tax levels Federal tax reform ( show ) Automated payment transaction tax 9 -- 9 -- 9
Competitive Tax Plan Efficient Taxation of Income FairTax Flat tax Hall -- Rabushka flat tax Kemp Commission Taxpayer Choice Act USA Tax Value added tax Border
- adjustment tax United States portal Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters
. The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties
specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
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
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 5>

<item 6>
IRS penalties
Some of these taxes require purchase of tax stamps or other evidence of advance payment of tax . Some require collection of the tax by retailers . An assortment
of penalties apply to manufacturers and retailers not complying with the particular rules . In addition , certain penalties are imposed in the form of an excise
tax . Pension and benefit plans must pay a tax for a variety of failures . Charities and private foundations must pay an excise tax on prohibited transactions
and other failures . Tax fraud penalties ( edit ) Intentional filing of materially false tax returns is a criminal offence . A person convicted of committing
tax fraud , or aiding and abetting another in committing tax fraud , may be subject to forfeiture of property and / or jail time . Conviction and sentencing is
through the court system . Responsibility for prosecution falls to the U.S. Department of Justice , not the Internal Revenue Service . Penalties may be assessed
against tax protesters who raise arguments that income tax laws are not valid , or who otherwise file frivolous returns or court petitions .
</item 6>

<item 7>
IRS penalties
Tax adviser penalties ( edit ) Penalties also apply to people who promote tax shelters or who fail to maintain and disclose lists of reportable transactions
their customers or clients for those transactions . These monetary penalties can be severe . Judicial appeal of penalties ( edit ) Most penalties are subject to
judicial review . However , the courts rarely modify assessment of the penalties and interest for underestimate or late payment . Further reading ( edit )
Meldman , Robert E. , Federal Taxation Practice and Procedure , ISBN 978 - 0 - 8080 - 1647 - 2 , Chapters 9 and 18 Shafiroff , Ira L. , Internal Revenue Service
Practice and Procedure Deskbook , Kindle edition ( no ISBN ) , or hardcover ISBN 978 - 0 - 87224 - 109 - 1 Pratt , Kulsrud , et al , Federal Taxation , 2006
Edition ISBN 978 - 0 - 7593 - 5175 - 2 , pages 2 -- 6 through 2 - 15 .
</item 7>

<item 8>
IRS penalties
Willis , Hoffman , et al , South - Western Federal Taxation , 2009 Edition ISBN 978 - 0 - 324 - 66052 - 4 , pages 26 -- 13 through 26 - 19 . References ( edit )
Jump up ^ Most of the penalty rules are in Chapters 68 through 75 of the Internal Revenue Code , sections 6651 through 7344 . 26 USC Subtitle F . Jump up ^ Alan
J. Tarr , J.D. , LL. M. ( Taxation ) & Pamela Jensen Drucker , J.D. , LL. M. ( Taxation ) , Civil Tax Penalties , U.S. Income Portfolios , Vol. 634 ( 2d ed.
2012 ) , Bloomberg BNA . Jump up ^ 26 USC 6654 . Jump up ^ 26 USC 6651 ( a ) ( 3 ) . Jump up ^ See IRS Tax Topic 653 . Jump up ^ 26 USC 6651 ( a ) ( 1 ) .
</item 8>

<item 9>
IRS penalties
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 9>

<item 10>
IRS penalties
Penalties may be monetary or may involve forfeiture of property . Criminal penalties may include jail time , but are imposed only by a federal judge after a
defendant is convicted . Most monetary penalties are based on the amount of tax not properly paid . Penalties may increase with the period of nonpayment . Some
penalties are fixed dollar amounts or fixed percentages of some measure required to be reported . Excise taxes used as penalties are imposed in the Code
sections relating to particular kinds of transactions . Some penalties may be waived or abated where the taxpayer shows reasonable cause for the failure .
Penalties apply for failures to file income tax returns or information returns , or for filing incorrect returns . Some penalties may be very minor . Penalties
apply for certain types of errors on tax returns , and may be substantial . Some penalties are imposed as excise taxes on particular transactions . Certain
other penalties apply for other types of failures . Certain acts may result in forfeiture of property of the taxpayer . There are over 150 kinds of civil
penalties in the U.S. Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines .
</item 10>
=== END ITEM id=01db2bc0fe7b2832 ===

=== ITEM id=45567a36dd94280a ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.
  Fact 0.1: Beijing was formerly romanized as Peking.
  Fact 0.2: Beijing is the world's second most populous city proper.
  Fact 0.3: Beijing is the most populous capital city.
  Fact 0.4: Beijing is located in northern China.
  Fact 0.5: Beijing is governed as a direct-controlled municipality under the national government.
  Fact 0.6: Beijing has 16 urban, suburban, and rural districts.
  Fact 0.7: Beijing Municipality is surrounded by Hebei Province, except for the neighboring Tianjin Municipality to the southeast.
  Fact 0.8: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the Jingjinji metropolitan region.
  Fact 0.9: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the national capital region of China.
Sentence 1: Over the past 3,000 years , the city of Beijing has had numerous other names . The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University , still use the former romanization .
  Fact 1.0: Over the past 3,000 years, the city of Beijing has had numerous other names.
  Fact 1.1: The name Beijing means 'Northern Capital'.
  Fact 1.2: The name Beijing is made of the Chinese characters 北 (north) and 京 (capital).
  Fact 1.3: The name Beijing was applied to the city in 1403 during the Ming dynasty.
  Fact 1.4: The name Beijing was applied to distinguish the city from Nanjing, the 'Southern Capital'.
  Fact 1.5: The English spelling Beijing is based on the pinyin romanization of the two characters as pronounced in Standard Mandarin.
  Fact 1.6: The older English spelling Peking is the postal romanization of the same two characters as pronounced in Chinese dialects spoken in the southern port towns.
  Fact 1.7: The southern port towns were the first visited by European traders and missionaries.
  Fact 1.8: The Chinese dialects spoken in the southern port towns preserve the Middle Chinese pronunciation of 京 as kjaeng.
  Fact 1.9: That pronunciation of 京 as kjaeng comes before a phonetic shift in the northern dialects to the modern pronunciation.
  Fact 1.10: Peking is no longer the common name for the city.
  Fact 1.11: Some of the city's older locations and facilities still use the former romanization Peking.
  Fact 1.12: Beijing Capital International Airport, with IATA code PEK, still uses the former romanization.
  Fact 1.13: Peking University still uses the former romanization.

Retrieved items:
<item 1>
Beijing
Two generations later , Kublai Khan ordered the construction of Dadu ( or Daidu to the Mongols , commonly known as Khanbaliq ) , a new capital for his Yuan
dynasty to the northeast of the Zhongdu ruins . The construction took from 1264 to 1293 , but greatly enhanced the status of a city on the northern fringe of
China proper . The city was centered on the Drum Tower slightly to the north of modern Beijing and stretched from the present - day Chang'an Avenue to the
northern part of Line 10 subway . Remnants of the Yuan rammed earth wall still stand and are known as the Tucheng . Ming dynasty One of the corner towers of the
Forbidden City , which was built by Emperor Yongle in Ming dynasty . In 1368 , soon after declaring the new Hongwu era of the Ming dynasty , the rebel leader
Zhu Yuanzhang sent an army to Khanbaliq and conquered it . Since the Yuan continued to occupy Shangdu and Mongolia , however , Dadu was renamed to Beiping as
used to supply the military garrisons in the area .
</item 1>

<item 2>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 2>

<item 3>
Beijing
tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and
most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16
urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the
southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China . As a city combining both modern and
traditional architecture , Beijing is a megacity rich in history , exemplified in its global influence in politics , economy , education , history , culture ,
and technology . Beijing is the second largest Chinese city by urban population after Shanghai and is the nation 's political , cultural , and educational
center . It is home to the headquarters of most of China 's largest state - owned companies and is a major hub for the national highway , expressway , railway ,
and high - speed rail networks .
</item 3>

<item 4>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 4>

<item 5>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 5>

<item 6>
Beijing
1 Towns 4.1. 2 Neighborhoods 4.2 Judiciary and procuracy 4.3 Diplomatic missions 5 Economy 5.1 Sector composition 5.2 Economic zones 6 Demographics 6.1
Metropolitan area 7 Culture 7.1 Places of interest 8 Religion 8.1 Chinese folk religion and Taoism 8.2 Buddhism 8.3 Islam 8.4 Christianity 9 Media 9.1
Television and radio 9.2 Press 10 Sports 10.1 Events 10.2 Venues 10.3 Clubs 11 Transportation 11.1 Rail and high - speed rail 11.2 Roads and expressways 11.3
Air 11.4 Public transit 11.5 Taxi 11.6 Bicycles 12 Defense and aerospace 13 Nature and wildlife 14 See also 15 References 16 Further reading 17 External links
Etymology See also : Names of Beijing Over the past 3,000 years , the city of Beijing has had numerous other names .
</item 6>

<item 7>
Beijing
In 938 , after the fall of the Tang , the Later Jin ceded the entire northern frontier to the Khitan Liao dynasty , which treated the city as Nanjing , or the
`` Southern Capital '' , one of four secondary capitals to complement its `` Supreme Capital '' , Shangjing ( modern Baarin Left Banner in Inner Mongolia ) .
Some of the oldest surviving structures in Beijing date to the Liao period , including the Tianning Pagoda . Miaoying Temple and its White Stupa in Dadu which
was built by Kublai Khan The Liao fell to the Jurchen Jin dynasty in 1122 , which gave the city to the Song dynasty and then retook it in 1125 during its
conquest of northern China . In 1153 , the Jurchen Jin made Beijing their `` Central Capital '' , or Zhongdu . The city was besieged by Genghis Khan 's invading
Mongolian army in 1213 and razed to the ground two years later .
</item 7>

<item 8>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 8>

<item 9>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 9>

<item 10>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 10>
=== END ITEM id=45567a36dd94280a ===

=== ITEM id=9758262d5fb6b2bf ===
Gold sentences and facts:
Sentence 0: The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of the Mahanadi Delta .
  Fact 0.0: The Eastern Coastal Plains refers to a wide stretch of landmass of India.
  Fact 0.1: The Eastern Coastal Plains lie between the Eastern Ghats and the Bay of Bengal.
  Fact 0.2: The Eastern Coastal Plains are wider and more level than the western coastal plains.
  Fact 0.3: The Eastern Coastal Plains stretch from Tamil Nadu in the south to West Bengal in the north.
  Fact 0.4: The Eastern Coastal Plains run through Andhra Pradesh and Odisha.
  Fact 0.5: Chilka Lake is a brackish water lake.
  Fact 0.6: Chilka Lake lies along the eastern coastal plain.
  Fact 0.7: Chilka Lake lies in the state of Odisha.
  Fact 0.8: Chilka Lake stretches to the south of the Mahanadi Delta.

Retrieved items:
<item 1>
Sub-Saharan Africa
Arabian Peninsula Najd Hejaz Tihamah Eastern Arabia South Arabia Hadhramaut Arabian Peninsula coastal fog desert Tigris -- Euphrates Mesopotamia Upper
Mesopotamia Lower Mesopotamia Sawad Nineveh plains Akkad ( region ) Babylonia Canaan Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant
Southern Levant Transjordan Jordan Rift Valley Israel Levantine Sea Golan Heights Hula Valley Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai
Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater
Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt
South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir
Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal
Plains Meghalaya subtropical forests MENASA Lower
</item 1>

<item 2>
Eastern coastal plains
The width of the plains varies between 100 and 130 km ( 62 to 80 miles ) . It is locally known as Utkal Plains in the Northern part between Cossye and
Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of
river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains . See also ( edit ) Western Coastal Plains
References ( edit ) Jump up ^ Sr. Bimcy ; Sr. Sisily ; Charlotte . Bibliographic information . Scholar Publishing House . pp. 20 -- 21 . ISBN 8171725163 .
</item 2>

<item 3>
Geography of Nigeria
Plains rise to the north of the valleys . To the southwest of the Niger there is `` rugged '' highland , and to the southeast of the Benue hills and mountains
are found all the way to the border with Cameroon . Coastal plains are found in both the southwest and the southeast . Niger Delta ( edit ) This section needs
expansion . You can help by adding to it . ( January 2011 ) The Niger Delta is located in the southern part of Nigeria . It is one of the world 's largest
arcuate fan - shaped river deltas . The riverine area of the Niger Delta is a coastal belt of swamps bordering the Atlantic ocean . The swamps are vegetated
tidal flats formed by a reticulate pattern of interconnected meandering creeks and distributaries of the River Niger . Vegetation of the Niger Delta ( edit )
The vegetation of the Niger Delta consist mainly of forest swamps . The forest are of two types , nearest the sea is a belt of saline / brackish Mangrove swamp
separated from the sea by sand beach ridges within the mangrove swamp . Numerous sandy islands occur with fresh water vegetation .
</item 3>

<item 4>
Eastern coastal plains
pp. 20 -- 21 . ISBN 8171725163 . Geography of India Climate Climate Climatic regions Geology Fossil Parks Geology of India Indian Plate Stones Landforms Beaches
Canals Desert Extreme points Glaciers Islands Lakes Mountains Rivers Valleys Volcanoes Waterfalls Plains Indo - Gangetic Eastern Coastal Western Coastal Regions
North India Northeast India East India South India West India Central India Subdivisions Autonomous administrative divisions States and territories Districts
Cities Towns Municipalities Environment Biogeographic classification Ecoregions Fauna Flora Forests Issues Protected Areas Wildlife This Indian location article
is a stub . You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Eastern_coastal_plains&oldid=812520937 ''
Categories : Plains of India Coasts of India India geography stubs Hidden categories : Articles needing additional references from May 2015 All articles needing
additional references India articles missing geocoordinate data All articles needing coordinates All stub articles Talk About Wikipedia हिन्दी Português தமிழ்
Edit links This page was last edited on 28 November 2017 , at 08 : 32 .
</item 4>

<item 5>
Ring of Fire
Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant Southern Levant Transjordan Jordan Rift Valley Levantine Sea Golan Heights Hula Valley
Gaza Strip West Bank Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria
Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland
Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats
Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus
Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous
forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False
Divi Point Hindi Belt Ladakh Aksai Chin
</item 5>

<item 6>
India’s Eastern Coastal Plains lie between the Eastern Ghats and Bay of Bengal, from Tamil Nadu to West Bengal. About 100–130 km wide, they include the Utkal
Plains, Northern Circars and Coromandel Coast. The Mahanadi, Godavari, Krishna and Kaveri form major deltas; rainfall comes from both monsoons, and Odisha’s
Chilika Lake is a brackish lagoon.

Since independence, Indian place names have changed through local
</item 6>

<item 7>
Ancient Indian agriculture included a Sanskrit classification of 12 land types, from fertile and barren to river-watered and rainfed. Rice may have been
domesticated near the Ganges by the sixth millennium BC; northwest India cultivated cereals and legumes even earlier. Evidence from 12 southern sites shows
diverse crops, including pulses, millets, wheat, barley, cotton and linseed. Some accounts trace agriculture to 9000 BC, followed by settled life and improved
tools. Double monso
</item 7>

<item 8>
Dead Sea
Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu
Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley
Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic
plains moist deciduous forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau
Coromandel Coast Konkan False Divi Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay
of Bengal Gulf of Khambhat Gulf of Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman
Islands Nicobar Islands Maldive Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater
</item 8>

<item 9>
India’s Eastern Coastal Plains stretch from Tamil Nadu to West Bengal, with major river deltas and Chilika Lagoon. Agriculture remains vital despite its
declining GDP share: it employs roughly half the workforce, supports major exports, and ranks among the world’s largest producers of crops, milk, and fish. The
Green Revolution achieved food self-sufficiency, while horticulture and fisheries expanded. However, yields remain below potential, and small farms, monsoon
dependence, weak irrigation, infrastructure, storage,
</item 9>

<item 10>
Eastern coastal plains
Eastern coastal plains - Wikipedia Eastern coastal plains Jump to : navigation , search This article relies largely or entirely on a single source . Relevant
discussion may be found on the talk page . Please help improve this article by introducing citations to additional sources . ( May 2015 ) View of Fields at
Biccavolu , Eastern coastal plains , Andhra Pradesh The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats
and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through
Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of
the Mahanadi Delta . Deltas of many of India 's rivers form a major portion of these plains . The Mahanadi , Godavari , Kaveri and Krishna rivers drain these
plains . The region receives both the Northeast & Southwest monsoon rains with its annual rainfall averaging between 1,000 and 3,000 mm ( 39 and 118 in ) .
</item 10>
=== END ITEM id=9758262d5fb6b2bf ===

=== ITEM id=b5b82483de76e126 ===
Gold sentences and facts:
Sentence 0: A third series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be set in 1937 .
  Fact 0.0: A third series was confirmed to be in production by writer Simon Nye.
  Fact 0.1: Simon Nye confirmed the third series at the BFI and Radio Times Television Festival on 8 April 2017.
  Fact 0.2: Simon Nye described the third series as having some exotic new animals.
  Fact 0.3: Simon Nye said production of the third series would begin in three weeks' time.
  Fact 0.4: Production of the third series would begin upon Keeley Hawes's arrival in Corfu for filming.
  Fact 0.5: The third series will be set in 1937.

Retrieved items:
<item 1>
The Durrells
Jump up ^ Huntman , Ruth ( 26 March 2016 ) . `` Gerald Durrell was my hero ... I married him for his zoo '' . The Guardian . Retrieved April 26 , 2017 . Jump up
^ `` SUNNIER CLIMES . Where is The Durrells filmed ? Corfu locations and Mediterranean settings for the ITV drama '' . The Sun . 28 May 2017 . Retrieved 27
December 2017 . Jump up ^ Travis , Ben ( 23 April 2017 ) . `` The Durrells , Series 2 : cast , locations , and three other things to know as the Keeley Hawes
drama returns '' . London Evening Standard . Retrieved 27 December 2017 . ^ Jump up to : `` BARB - Weekly Top 10 Programmes '' . Broadcasters ' Audience
Research Board . Retrieved 9 May 2016 . Jump up ^ `` BARB - Weekly Top 30 Programmes '' . Broadcasters ' Audience Research Board . Retrieved 9 May 2016 .
</item 1>

<item 2>
The Durrells
The Durrells - wikipedia The Durrells Jump to : navigation , search The Durrells Genre Comedy - drama Based on The Corfu Trilogy by Gerald Durrell Written by
Simon Nye Theme music composer Ruth Barrett Country of origin United Kingdom Original language ( s ) English No. of series No. of episodes 12 Production
Executive producer ( s ) Lee Morris Sally Woodward Gentle Simon Nye Producer ( s ) Christopher Hall Cinematography Julian Court , James Aspinall Running time 46
-- 47 minutes Production company ( s ) Sid Gentle Films Masterpiece co-production Release Original network ITV Picture format 16 : 9 1080i Audio format Stereo
Original release 3 April 2016 ( 2016 - 04 - 03 ) -- present External links Website The Durrells ( also known as The Durrells in Corfu on American television )
is a British comedy - drama series based on Gerald Durrell 's three autobiographical books about his family 's four years ( 1935 -- 1939 ) on the Greek Island
of Corfu , which began airing on 3 April 2016 .
</item 2>

<item 3>
The Durrells
The show and its episodes attained high ratings on IMDb , around 8 / 10 . BROADCAST ( edit ) Internationally , the series was acquired in Australia by the Seven
Network and premiered on 24 August 2016 . In the United States , PBS began airing the show , retitled as The Durrells in Corfu , on 16 October 2016 at 8pm . The
Durrells started screening in New Zealand on 26 October 2016 on Prime TV . In Canada , the Canadian Broadcasting Corporation began airing Series 1 on 13
September 2017 . In Spain , the series has been acquired by the streaming platform Filmin . A third series of The Durrells was announced by writer Simon Nye at
the BFI and Radio Times Television Festival on 8 April 2017 . It began filming in May 2017 and is expected to air on ITV in the Spring of 2018 .
</item 3>

<item 4>
The Durrells
Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 - 04 - 03 ) 8 May 2016 ( 2016 - 05 - 08 ) 6 23 April 2017 ( 2017 - 04 - 23 ) 28 May 2017 (
2017 - 05 - 28 ) Episodes ( edit ) Series 1 ( 2016 ) ( edit ) ( 1935 ) No . overall No. in series Title Directed by Written by Original air date UK viewers (
millions ) `` Episode 1 '' Steve Barron Simon Nye 3 April 2016 ( 2016 - 04 - 03 ) 8.81 The island of Corfu 's distinct contrast with England is what enthrals
Bournemouth mother Louisa Durrell to uproot her disenchanted family from a life of no prospects and venture into one with a lessened need for conformity .
However , the appeal becomes ever distant to see upon their arrival .
</item 4>

<item 5>
The Durrells
Jump up ^ `` Twitter - ITV Media '' . Twitter . Retrieved 11 April 2016 . Jump up ^ `` Digital Spy - Forums - The Ratings Thread ( Part 67 ) '' . Digital Spy
Forums . Retrieved 11 April 2016 . Jump up ^ `` BARB - Weekly Top 10 Programmes ( w / e 3rd April 2016 ) '' . Broadcasters ' Audience Research Board . Retrieved
11 April 2016 . Jump up ^ `` ITV Media - Press Releases - `` ITV recommissions hit drama The Durrells '' `` . ITV Media . Retrieved 16 April 2016 . Jump up ^ ``
ITV Press Centre - `` More Sun , Sea and Animals as the cast of The Durrells return to Corfu to film the second series of the ITV drama '' `` . ITV Press Centre
. ITV . Retrieved 12 August 2016 . Jump up ^ `` IMDb - The Durrells '' . Internet Movie Database . Retrieved 12 August 2016 .
</item 5>

<item 6>
The Durrells
See also ( edit ) Gerald Durrell 's Corfu trilogy books : My Family and Other Animals ( 1956 ) Birds , Beasts , and Relatives ( 1969 ) The Garden of the Gods (
1978 ) My Family and Other Animals , a 1987 BBC series based on the Corfu trilogy by Durrell My Family and Other Animals , a 2005 BBC telemovie based on the
1956 book by Durrell References ( edit ) Jump up ^ `` Early life of Gerald Durrell comes to TV screens soon '' . Jersey Evening Post . 24 March 2016 . Retrieved
1 April 2016 . Jump up ^ `` ITV commissions new six - part drama The Durrells '' . ITV Press Centre . 18 June 2015 . Retrieved 1 April 2016 . Jump up ^ `` The
Durrells ( TV Series 2016 -- ) '' . Internet Movie Database . Retrieved 1 April 2016 . Jump up ^ Huntman , Ruth ( 26 March 2016 ) .
</item 6>

<item 7>
The Durrells
Jump up ^ The episode 's world premiere date was at the BFI and Radio Times Television Festival on 8 April 2017 . Series 3 ( 2018 ) ( edit ) ( 1937 ) A third
series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as
having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be
set in 1937 . Reception ( edit ) Reception to the first episode was positive , with Gerard O'Donovan ( The Telegraph ) calling it `` a series that 's not only
sun - drenched and liberating , but also catches its source material 's high good humour without labouring it and weaves an authentic sense of the innocent
exoticism of the original , '' before awarding it four stars .
</item 7>

<item 8>
The Durrells
The series is written by Simon Nye , directed by Steve Barron and Roger Goldby , and produced by Christopher Hall . The executive producers are Lee Morris and
Sally Woodward Gentle. Lee Durrell , Gerald Durrell 's widow and director of the Durrell Wildlife Conservation Trust , acted as consultant . The series is
partly filmed on Corfu and at Ealing Studios in London . Contents ( hide ) 1 Story 2 Characters 3 Series overview 4 Episodes 4.1 Series 1 ( 2016 ) 4.2 Series 2
( 2017 ) 4.3 Series 3 ( 2018 ) 5 Reception 6 Broadcast 7 See also 8 References 9 External links Story ( edit ) The series begins in 1935 , when Louisa Durrell
suddenly announces that she and her four children will move from Bournemouth to the Greek island of Corfu . Her husband has died some years earlier and the
family is experiencing financial problems . A Homeric battle ensues as the family adapts to life on the island which , despite a lack of electricity , is cheap
and an earthly paradise .
</item 8>

<item 9>
The Durrells
Internet Movie Database . Retrieved 12 August 2016 . Jump up ^ Knox , David ( 30 June 2016 ) . `` Seven picks up UK drama The Durrells '' . TV Tonight .
Retrieved 27 August 2016 . Jump up ^ Dunk , Tiffany ( 24 August 2016 ) . `` Seven takes punt that The Durrells can recreate Downton Abbey 's huge success '' .
news.com.au . Retrieved 27 August 2016 . Jump up ^ `` Screener - TV Listings - `` The Durrells in Corfu on Masterpiece '' `` . 8 October 2016 . Retrieved 8
October 2016 . Jump up ^ `` CBC ANNOUNCES FALL 2017 BROADCAST PREMIERE DATES FOR NEW AND RETURNING TELEVISION SERIES '' . CBC Media Centre . Jump up ^ `` Los
Durrell '' . Retrieved 27 September 2017 . Jump up ^ 28 day data : 7,247,000 on ITV , 1,218,000 on ITV HD and 343,
</item 9>

<item 10>
The Durrells
Characters ( edit ) Main Keeley Hawes as Louisa Durrell Milo Parker as Gerry Durrell Josh O'Connor as Larry Durrell Daisy Waterstone as Margo Durrell Callum
Woodhouse as Leslie Durrell Lucy Black as Florence Petrides Yorgos Karamihos as Dr. Theo Stephanides Alexis Georgoulis as Spiros Halikiopoulos Leslie Caron as
Countess Mavrodaki Ulric von der Esch as Sven Lundblad Anna Savva as Lugaretzia Errika Bigiou as Vasilia ( series 2 ) Recurring James Cosmo as Captain Creech
Maximilian Befort as Max Manolis Emmanouel as Sotos Andrew Bicknell as Headmaster Graham Seed as Mr Trevitt Liz Watts as Nancy Ben Hall as Donald Hara - Joy
Ermidi as Alexia Yorgos Tryfonas as Market Trader Nick Orestis Chaniotakis as Monk Arsenis Grimmas as Guest House Manager Spiros Kasfikis as Durrell 's House
Owner Daniel Lapaine as Hugh Jarvis Series overview ( edit ) Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 -
</item 10>
=== END ITEM id=b5b82483de76e126 ===

=== ITEM id=17adf7bf4cd3296e ===
Gold sentences and facts:
Sentence 0: Capital punishment in Australia has been abolished in all jurisdictions . Queensland abolished the death penalty in 1922 . Tasmania did the same in 1968 , the federal government abolished the death penalty in 1973 , with application also in the Australian Capital Territory and the Northern Territory . Victoria did so in 1975 , South Australia in 1976 , and Western Australia in 1984 . New South Wales abolished the death penalty for murder in 1955 , and for all crimes in 1985 . In 2010 , the federal government passed legislation prohibiting the re-establishment of capital punishment by any state or territory . Neither the Commonwealth nor any of the states will extradite or deport a prisoner to another jurisdiction if they will face the death penalty , and police co-operation with other countries which have the death penalty has been questioned .
  Fact 0.0: Capital punishment in Australia has been abolished in all jurisdictions.
  Fact 0.1: Queensland abolished the death penalty in 1922.
  Fact 0.2: Tasmania abolished the death penalty in 1968.
  Fact 0.3: The federal government abolished the death penalty in 1973.
  Fact 0.4: The federal abolition of the death penalty in 1973 applied in the Australian Capital Territory.
  Fact 0.5: The federal abolition of the death penalty in 1973 applied in the Northern Territory.
  Fact 0.6: Victoria abolished the death penalty in 1975.
  Fact 0.7: South Australia abolished the death penalty in 1976.
  Fact 0.8: Western Australia abolished the death penalty in 1984.
  Fact 0.9: New South Wales abolished the death penalty for murder in 1955.
  Fact 0.10: New South Wales abolished the death penalty for all crimes in 1985.
  Fact 0.11: In 2010, the federal government passed legislation prohibiting the re-establishment of capital punishment by any state or territory.
  Fact 0.12: The Commonwealth will not extradite a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.13: No state will extradite a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.14: The Commonwealth will not deport a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.15: No state will deport a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.16: Police co-operation with other countries which have the death penalty has been questioned.

Retrieved items:
<item 1>
Capital punishment in Australia
Retrieved 24 April 2017 . Jump up ^ `` Small majority of Australians favour the death penalty for deadly terrorist acts in Australia '' . Roy Morgan . Melbourne
, Vic. : Roy Morgan Research Ltd. 19 September 2014 . Retrieved 3 December 2017 . Sources ^ Death Penalty Abolition Act 1973 Capital Punishment from the
Fremantle Prison site Trends & Issues in Crime and Criminal Justice No. 3 : Capital punishment from the Australian Institute of Criminology Dhakiyarr Wirrpanda
-- Appeal for Justice `` Capital Punishment '' speech by The Ian Callinan , Justice of the High Court of Australia The History of Correctional Services in South
Australia External links ( edit ) `` List of Executions in the State of Victoria '' . Archived from the original on 8 October 2011 . `` The Death Penalty (
Queensland ) '' . Life & Death in the Sunshine State . Archived from the original on 11 May 2015 .
</item 1>

<item 2>
Capital punishment in Australia
Neither the Commonwealth nor any of the states will extradite or deport a prisoner to another jurisdiction if they will face the death penalty . A recent case
involving this was the case of American Gabe Watson , who was convicted of the manslaughter of his wife in North Queensland , and faced capital murder charges
in his home state of Alabama . His deportation was delayed until the government received assurances that he would not be executed if found guilty . New South
Wales ( edit ) The last execution in New South Wales was carried out on 24 August 1939 , when John Trevor Kelly was hanged at Sydney 's Long Bay Correctional
Centre for the murder of Marjorie Constance Sommarlad . New South Wales abolished the death penalty for murder in 1955 , but retained it as a potential penalty
for treason , piracy , and arson in naval dockyards until 1985 . New South Wales was the last Australian State to formally abolish the death penalty for all
crimes .
</item 2>

<item 3>
Capital punishment in Australia
Northern Territory ( edit ) Aborigines who lived in the Northern Territory came under the European law of Australia even though they did not have any contact
with the government . There were several outcries over Aborigines receiving mandatory death sentences for murder , leading to the passage of the Crimes
Ordinance 1934 which allowed for discretionary sentences when both the accused and the victim were Aboriginal . The last execution was a double hanging in 1952
, and the death penalty was abolished 1973 . Public opinion ( edit ) In Australia capital punishment was banned on a state - by - state basis through the 20th
century . Despite the ban , polls have indicated varying support for the reintroduction of the practice . A 2005 Bulletin poll showed that most Australians
supported capital punishment . The Australian National University 's 2007 Electoral Survey found that 44 per cent of people thought the death penalty should be
reintroduced , while only 38 per cent disagreed . In the recent case of the Bali bombers , then prime minister John Howard stated that Australians expected
their execution by Indonesia . On occasion the issue of capital punishment is published in the media or is subject to media and public support and scrutiny .
</item 3>

<item 4>
Capital punishment in Australia
Capital punishment in Australia - wikipedia Capital punishment in Australia Jump to : navigation , search Capital punishment in Australia has been abolished in
all jurisdictions . Queensland abolished the death penalty in 1922 . Tasmania did the same in 1968 , the federal government abolished the death penalty in 1973
, with application also in the Australian Capital Territory and the Northern Territory . Victoria did so in 1975 , South Australia in 1976 , and Western
Australia in 1984 . New South Wales abolished the death penalty for murder in 1955 , and for all crimes in 1985 . In 2010 , the federal government passed
legislation prohibiting the re-establishment of capital punishment by any state or territory . Neither the Commonwealth nor any of the states will extradite or
deport a prisoner to another jurisdiction if they will face the death penalty , and police co-operation with other countries which have the death penalty has
been questioned . The last execution in Australia took place in 1967 , when Ronald Ryan was hanged in Victoria . Between Ryan 's execution in 1967 and 1984 ,
several more people were sentenced to death , but had their sentences commuted to life imprisonment .
</item 4>

<item 5>
Capital punishment in Australia
New South Wales was the last Australian State to formally abolish the death penalty for all crimes . Victoria ( edit ) Old Melbourne Gaol gallows Victoria 's
first executions occurred in 1842 when two Aboriginal men , Tunnerminnerwait and Maulboyheenner , were hanged outside the site of the Melbourne Gaol for the
killing of two whalers in the Westernport district . Ronald Ryan was the last man executed at Pentridge Prison and in Australia . Ryan was hanged on 3 February
1967 after being convicted of the shooting death of a prison officer during a prison escape from Pentridge Prison , Coburg , Victoria in 1965 . Ryan was the
last of 186 executions . Victoria was also the state of the last woman executed in Australia . Jean Lee was hanged in 1951 . She was accused of being an
accomplice in the murder of 73 - year - old William ( ' Pop ' ) Kent . She , along with her accomplices were executed on 19 February 1951 .
</item 5>

<item 6>
Capital punishment in Australia
The last sentence of death in Western Australia ( indeed anywhere in Australia ) was passed in 1984 , but the female killer ( Brenda Hodge ) in question had her
sentence commuted to imprisonment for life , as was customary by this stage . Capital punishment was formally removed from the statutes of the state with the
passage of the Acts Amendment ( Abolition of Capital Punishment ) Act 1984 . South Australia ( edit ) The Adelaide Gaol was the site of forty - four hangings ,
from Joseph Stagg on 18 November 1840 to Glen Sabre Valance , murderer and rapist , on 24 November 1964 and three executions also occurred at Mount Gambier Gaol
. Two Ngarrindjeri men were controversially executed by hanging along the Coorong on 22 August 1840 , after a drumhead court - martial conducted by Police
Commissioner O'Halloran on the orders of Governor George Gawler . The men were found to be guilty of murdering the twenty - five survivors of the shipwreck
Maria a few months before .
</item 6>

<item 7>
Capital punishment in Australia
Most occasions where capital punishment is brought up in the media , it is regarding current cases of intense media coverage regarding murder , rape and in
extreme circumstances such as terrorism . On various occasions , the media and public express support for capital punishment for the most heinous of crimes
including mass murder such as in the cases of the Milat Backpacker Murders and the Bryant Port Arthur massacre , in which a total of 42 people were killed
stirring strong emotions as to whether or not to reintroduce the death penalty . A similar surge in public support for reintroduction occurred when Lindy
Chamberlain was convicted of murdering her baby Azaria in 1982 , although it was subsequently shown that she was not guilty . This miscarriage of justice has
been used as a powerful example of the risks of the death penalty . However , no person of significant stature or influence has advocated the death penalty for
quite some time since the last execution in 1967 . The death penalty was completely abolished in Australia with the Crimes Legislation Amendment ( Torture
Prohibition and Death Penalty Abolition ) Bill 2009 passing the Australian Senate without amendments in March 2010 .
</item 7>

<item 8>
Capital punishment in Australia
The last death sentence was given in August 1984 , when Brenda Hodge was sentenced to death by Western Australia ( and subsequently had her sentence commuted to
life imprisonment ) . Contents ( hide ) 1 History 1.1 Commonwealth 1.2 New South Wales 1.3 Victoria 1.4 Queensland 1.5 Western Australia 1.6 South Australia 1.7
Tasmania 1.8 Australian Capital Territory 1.9 Northern Territory 2 Public opinion 3 References 4 External links History ( edit ) Death sentences were carried
out under Aboriginal customary law , either directly or through sorcery . In some cases the condemned could be denied mortuary rites . The first executions
carried out under European law in Australia took place in Western Australia in 1629 , when Dutch authorities hanged the mutineers of the Batavia . Capital
punishment had been part of the legal system of Australia since British settlement and during the 19th century , crimes that could carry a death sentence
included burglary , sheep stealing , forgery , sexual assaults , murder and manslaughter , and there is one reported case of someone being executed for `` being
illegally at large '' .
</item 8>

<item 9>
Capital punishment in Australia
During the 19th century , these crimes saw about 80 people hanged each year throughout Australia . Before and after federation , each state made its own
criminal laws and punishments . For a full list see : Main article : List of people legally executed in Australia Commonwealth ( edit ) In 1973 the Death
Penalty Abolition Act 1973 of the Commonwealth abolished the death penalty for federal offences . It provided in Section 3 that the Act applied to any offence
against a law of the Commonwealth , the Territories or under an Imperial Act , and in s . 4 that `` ( a ) person is not liable to the punishment of death for
any offence '' . No executions were carried out under the bridge of the federal government and the passage of the Death Penalty Abolition Act 1973 saw the death
penalty replaced with life imprisonment as their maximum punishment . Since the Commonwealth effects of utilising this Act no more individuals have been exposed
to the death penalty and it is now replaced with life imprisonment . On 11 March 2010 Federal Parliament passed laws that prevent the death penalty from being
reintroduced by any state or territory in Australia .
</item 9>

<item 10>
Capital punishment in Australia
Jump up ^ Helen MacDonald ( 2005 ) . Human remains : Dissection and Its Histories . Yale University Press . ISBN 978 - 0 - 300 - 11699 - 1 . Jump up ^ `` Two
executed '' . Northern Standard . Darwin , NT. 8 August 1952 . p. 1 . Retrieved 3 December 2017 . Jump up ^ Williams , George ( 2 March 2010 ) . `` No death
penalty , no shades of grey '' . Sydney Morning Herald . Retrieved 3 December 2017 . Jump up ^ Australians expect Bali bomber executions , says PM By Sarah
Smiles Canberra and Mark Forbes , 13 October 2007 , Brisbane Times Jump up ^ Death penalty dead and buried as Parliament bans it for good . Herald Sun ( 2010 -
03 - 12 ) . Jump up ^ `` Australians say penalty for murder should be Imprisonment ( 64 % ) rather than the Death Penalty ( 23 % ) '' , Roy Morgan Research , 27
August 2009 . Retrieved 24 April 2017 .
</item 10>
=== END ITEM id=17adf7bf4cd3296e ===
