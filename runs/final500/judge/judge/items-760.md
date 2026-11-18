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

=== ITEM id=4be6506d5af14e94 ===
Gold sentences and facts:
Sentence 0: The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at Newport Ranch , a resort and cattle ranch to the north of San Francisco . It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on May 4 , 2017 , and shows Brown and Alaina cruising down a road .
  Fact 0.0: The official music video for the song was directed by P.R. Brown.
  Fact 0.1: The official music video for the song was produced by Steve Lamar.
  Fact 0.2: The video was filmed along the Californian coast at The Inn at Newport Ranch.
  Fact 0.3: The Inn at Newport Ranch is a resort and cattle ranch.
  Fact 0.4: The Inn at Newport Ranch is north of San Francisco.
  Fact 0.5: The video features Brown and his duet partner Lauren Alaina singing along the coast.
  Fact 0.6: The video was released on May 14, 2017.
  Fact 0.7: An earlier lyric video was first released on May 4, 2017.
  Fact 0.8: The earlier lyric video shows Brown and Alaina cruising down a road.

Retrieved items:
<item 1>
What Ifs
Jump up ^ Bjorke , Matt ( January 3 , 2018 ) . `` Top 30 Country Digital Singles Chart : January 3 , 2018 '' . Roughstock . Retrieved January 14 , 2017 . ^ Jump
up to : Bjorke , Matt ( March 6 , 2018 ) . `` The Top 30 Digital Country Singles : March 5 , 2018 '' . Roughstock . Retrieved March 11 , 2018 . Jump up ^ ``
Watch Kane Brown , Lauren Alaina 's Picturesque ' What Ifs ' Video '' . Rolling Stone . May 15 , 2017 . Jump up ^ Kimble , Lindsay ( May 4 , 2017 ) . `` From
Middle School Chorus to Nashville , Kane Brown and Lauren Alaina Team Up for Duet ' What Ifs ' '' . Jump up ^ `` Kane Brown Chart History ( Canadian Hot 100 )
'' . Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard .
</item 1>

<item 2>
What Ifs
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
</item 2>

<item 3>
Graph query row:
work: official music video for What Ifs
relationship: LOCATED_IN
location: The Inn at Newport Ranch
</item 3>

<item 4>
What Ifs
Taste of Country . ^ Jump up to : Konicki , Lisa ( February 8 , 2017 ) . `` Kane Brown 's Current Single , `` What Ifs , '' With Lauren Alaina Almost Did n't
Make The Album `` . Nash Country Daily . Jump up ^ Freeman , Jon ( December 2 , 2016 ) . `` Kane Brown on Personal New Album , Why He Wo n't Sing First Hit :
Ram Report '' . Rolling Stone . ^ Jump up to : Asker , Jim ( October 17 , 2017 ) . `` Kane Brown Becomes First Artist To Simultaneously Lead Five Country Charts
'' . Billboard . Jump up ^ `` ' What Ifs ' sheet music '' . MusicNotes.com . Retrieved 5 February 2018 . Jump up ^ Bjorke , Matt ( December 13 , 2016 ) . `` Top
30 Digital Country Singles Chart : December 13 , 2016 '' . Roughstock . Retrieved February 5 , 2017 . Jump up ^ `` Country Airplay '' . Billboard .
</item 4>

<item 5>
What Ifs
wikipedia.org/w/index.php?title=What_Ifs&oldid=838232791 '' Categories : 2016 songs 2017 singles Kane Brown songs Lauren Alaina songs RCA Records Nashville
singles Vocal duets Song recordings produced by Dann Huff Songs written by Jordan Schmidt Billboard Hot Country Songs number - one singles Hidden categories :
Music infoboxes with deprecated parameters Articles with hAudio microformats Singlechart usages for Canada Singlechart called without song Singlechart usages
for Billboardcanadacountry Singlechart usages for Billboardhot100 Singlechart usages for Billboardcountryairplay Singlechart usages for Billboardcountrysongs
Certification Table Entry usages for Canada Certification Table Entry usages for United States Talk Contents About Wikipedia Español Edit links This page was
last edited on 25 April 2018 , at 18 : 59 . About Wikipedia
</item 5>

<item 6>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
What Ifs
What Ifs - wikipedia What Ifs Jump to : navigation , search `` What Ifs '' Single by Kane Brown featuring Lauren Alaina from the album Kane Brown Released
February 6 , 2017 ( 2017 - 02 - 06 ) Format Digital download Genre Country Length 3 : 08 Label RCA Nashville Songwriter ( s ) Kane Brown Matthew McGinn Jordan
Schmidt Producer ( s ) Dann Huff Kane Brown singles chronology `` Thunder in the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) `` Thunder in
the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) Lauren Alaina singles chronology `` Road Less Traveled '' ( 2016 ) Road Less Traveled 2016 ``
What Ifs '' ( 2017 ) What Ifs2017 `` Doin ' Fine '' ( 2017 ) Doin ' Fine 2017 `` What Ifs '' is a song recorded by American country music singer Kane Brown
featuring Lauren Alaina for his self - titled debut album .
</item 7>

<item 8>
What Ifs
The song is the first No. 1 on the chart for both Brown and Alaina . The following week , for charts dated October 28 , 2017 , `` What Ifs '' stayed at number
one , as well as topping the Country Airplay and Country Streaming Songs charts . Together with the release of the deluxe edition of the album which also
reached number on Top Country Albums , and another number one for a new track `` Heaven '' on digital sales , Brown became the first artist to have simultaneous
number ones on all five main country charts . The song was certified Platinum by the RIAA on October 2 , 2017 . It was the third most - consumed country song of
2017 , with 1,551,000 including streams ( 599,000 in pure sale ) sold in the year . It has sold 679,000 copies in the US as of March 2018 . Music video ( edit )
The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at
Newport Ranch , a resort and cattle ranch to the north of San Francisco .
</item 8>

<item 9>
What Ifs
The song was released with the album through RCA Nashville and was serviced to radio as the third single on February 6 , 2017 . Contents ( hide ) 1 Background 2
Composition 3 Commercial performance 4 Music video 5 Chart performance 5.1 Weekly charts 5.2 Year - end charts 6 Certifications and sales 7 References
Background ( edit ) The song was written by Brown with Matthew McGinn and Jordan Schmidt . The song is about a couple where the question `` what if '' is asked
repeatedly about what would happen in their relationship . It features a duet with Lauren Alaina , a school friend from Lakeview - Fort Oglethorpe High School
when he was living in Fort Oglethorpe , Georgia . According to Brown , the song was initially not intended to be a duet , he said : `` Once we decided on adding
a woman to `` What Ifs , '' I knew that I wanted Lauren to sing it , and , of course , she killed it . ''
</item 9>

<item 10>
What Ifs
According to Brown , another song was going to be used in the album , however , he was asked so much about `` What Ifs '' on social media that it was then
decided that the song should be included in the album . Composition ( edit ) The song is in the key of G minor with a moderately fast tempo of approximately 126
beats per minute . It follows the chord progression Gm - B ♭ - F. Brown 's vocals range from F - B ♭ . Commercial performance ( edit ) The song sold 6,000
downloads in the US the week the album was released on December 2 , 2016 , allowing it to chart at number 37 on the US Billboard Hot Country Songs chart . The
song first entered the US Billboard Country Airplay chart for the week of March 4 , 2017 at number 60 . It reached No. 1 on Hot Country Songs in October 2017 ,
ending the record breaking 34 - week reign of Sam Hunt 's `` Body Like a Back Road '' . The song is the first No. 1 on the chart for both Brown and Alaina .
</item 10>
=== END ITEM id=4be6506d5af14e94 ===

=== ITEM id=00e27f466eed575f ===
Gold sentences and facts:
Sentence 0: Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning , genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through artificial insemination .
  Fact 0.0: Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse.
  Fact 0.1: Superfecundation can lead to twin babies from two separate biological fathers.
  Fact 0.2: The term superfecundation is derived from fecund.
  Fact 0.3: Fecund means the ability to produce offspring.
  Fact 0.4: Heteropaternal superfecundation refers to the fertilization of two separate ova by two different fathers.
  Fact 0.5: Homopaternal superfecundation refers to the fertilization of two separate ova from the same father.
  Fact 0.6: Homopaternal superfecundation leads to fraternal twins.
  Fact 0.7: Heteropaternal superfecundation is referred to as a form of atypical twinning.
  Fact 0.8: Genetically, the twins from heteropaternal superfecundation are half siblings.
  Fact 0.9: Superfecundation is rare.
  Fact 0.10: Superfecundation can occur through separate occurrences of sexual intercourse.
  Fact 0.11: Superfecundation can occur through artificial insemination.

Retrieved items:
<item 1>
Superfecundation
Contents ( hide ) 1 Conception 2 Heteropaternal superfecundation in mammals 3 Cases involving superfecundation 4 See also 5 References 6 Further reading
Conception ( edit ) Superfecundation most commonly happens within hours or days of the first instance of fertilization with ova released during the same cycle .
The time window when eggs are able to be fertilized is small . Sperm cells can live inside a female 's body for four to five days . Once ovulation occurs , the
egg remains viable for 12 -- 48 hours before it begins to disintegrate . Thus , the fertile period can span five to seven days . Ovulation is usually suspended
during pregnancy to prevent further ova becoming fertilized and to help increase the chances of a full - term pregnancy . However , if an ovum is released after
the female was already impregnated when previously ovulating , a chance of a second pregnancy occurs , albeit at a different stage of development . This is
known as superfetation .
</item 1>

<item 2>
Superfecundation
This is known as superfetation . Heteropaternal superfecundation in mammals ( edit ) Heteropaternal superfecundation is common in animals such as cats and dogs
. Stray dogs can produce litters in which every puppy has a different sire . Though rare in humans , cases have been documented . In one study on humans , the
frequency was 2.4 % among dizygotic twins whose parents had been involved in paternity suits . Cases involving superfecundation ( edit ) Koen and Teun Stuart ,
Dutch boys who were the result of in vitro fertilization ( IVF ) . In a mixup at the laboratory , equipment had been used twice , causing another man 's sperm
to be mixed with the intended father 's . In 1995 , a young woman gave birth to diamniotic monochorionic twins , who were originally assumed to be monozygotic
twins until a paternity suit led to a DNA test . This led to the discovery that the twins had different fathers .
</item 2>

<item 3>
Superfecundation
This led to the discovery that the twins had different fathers . In 1982 , twins who were born with two different skin colors were discovered to be conceived as
a result of heteropaternal superfecundation . In 2001 , a case of spontaneous monopaternal superfecundation was reported after a woman undergoing IVF treatments
gave birth to quintuplets after only two embryos were implanted . Genetic testing supported that the twinning was not a result of the embryos splitting , and
that all five boys shared the same father . In 2016 , an IVF - implanted surrogate mother gave birth to two children : one genetically unrelated child from an
implanted embryo , and a biological child from her own egg and her husband 's sperm . See also ( edit ) Castor and Pollux , said to be twins having different
fathers Chimera ( genetics ) Mixed twins Polyspermy References ( edit ) Jump up ^ Worland , Justin ( May 8 , 2015 ) . `` The Science of How Women Can Have Twins
With 2 Different Fathers '' . Time . Retrieved March 23 , 2017 .
</item 3>

<item 4>
Superfecundation
PMID 1488855 . Jump up ^ Fierro , Pamela Prindle ( January 5 , 2016 ) , `` Heteropaternal Superfecundation - Twins with Different Fathers '' , About.com ,
archived from the original on April 4 , 2016 Jump up ^
https://www.scopus.com/record/display.uri?eid=2-s2.0-0020008620&origin=inward&txGid=F6227767DB3CDB724EA92EFC60A5A985.wsnAw8kcdt7IPYLO0V48gA%3a2 Jump up ^
Amsalem , Hagai ; Tsvieli , Rimona ; Zentner , Bat Sheva ; Yagel , Simcha ; Mitrani - Rosenbaum , Stella ; Hurwitz , Arye ( 2001 ) . `` Monopaternal
superfecundation of quintuplets after transfer of two embryos in an in vitro fertilization cycle '' . Fertility and Sterility. 76 ( 3 ) : 621 -- 3 .
</item 4>

<item 5>
Human fertilization
This possibility has been investigated by computer simulations of the fertilization process . See also ( edit ) Spontaneous conception , the unassisted
conception of a subsequent child after prior use of assisted reproductive technology References ( edit ) Jump up ^ Garrison , Fielding . An Introduction to the
History of Medicine , pages 566 - 567 ( Saunders 1921 ) . Jump up ^ `` Archived copy '' . Archived from the original on 2011 - 12 - 22 . Retrieved 2016 - 01 -
24 . Jump up ^ http://www.americanpregnancy.org/preventingpregnancy/pregnancyfaqmyths.html Jump up ^ Lawyers Guide to Forensic Medicine SBN 978 - 1 - 85941 -
159 - 9 By Bernard Knight - Page 188 `` Pregnancy is well known to occur from such external ejaculation ... '' Jump up ^ `` Fertilization of the Ovum '' . Gray
's Anatomy . Retrieved 2010 - 10 - 16 . Jump up ^ `` Fertilization '' . Retrieved 28 July 2010 .
</item 5>

<item 6>
Superfecundation
PMID 8064269 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Superfecundation&oldid=818981440 '' Categories : Reproduction Twin Triplets Multiple
births Free sex Sibling Hidden categories : Articles needing more detailed references All articles with dead external links Articles with dead external links
from June 2017 Articles needing additional references from May 2009 All articles needing additional references Talk Contents About Wikipedia Deutsch فارسی
Français Nederlands Polski Português Русский Slovenčina Edit links This page was last edited on 6 January 2018 , at 18 : 28 . About Wikipedia
</item 6>

<item 7>
Superfecundation
Superfecundation - wikipedia Superfecundation Jump to : navigation , search This article needs additional citations for verification . Please help improve this
article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2009 ) ( Learn how and when to remove this template
message ) Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to
twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal
superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two
separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning ,
genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through
artificial insemination .
</item 7>

<item 8>
Superfecundation
Fertility and Sterility. 76 ( 3 ) : 621 -- 3 . doi : 10.1016 / S0015 - 0282 ( 01 ) 01976 - 8 . PMID 11532493 . Jump up ^ Susan Scutti ( November 3 , 2017 ) . ``
Surrogate mom delivers two babies , one her own '' . CNN . Further reading ( edit ) `` Doctors Discover Dallas Twins Have Two Different Dads '' , Fox News , Sky
News , 2009 Lu , H.L. ; Wang , C.X. ; Wu , F.Q. ; Li , J.J. ( 1994 ) . `` Paternity identification in twins with different fathers '' . Journal of forensic
sciences . 39 ( 4 ) : 1100 -- 2 . PMID 8064269 .
</item 8>

<item 9>
Superfecundation
Time . Retrieved March 23 , 2017 . Jump up ^ `` Public Education Pamphlets '' . sogc.org . ^ Jump up to : Ambach , E ; Parson , W ; Brezinka , C ( 2000 ) . ``
Superfecundation and dual paternity in a twin pregnancy ending with placental abruption '' . Journal of forensic sciences . 45 ( 1 ) : 181 -- 3 . PMID 10641935
. ^ Jump up to : McNamara , Helen C. ; Kane , Stefan C. ; Craig , Jeffrey M. ; Short , Roger V. ; Umstad , Mark P. ( 2016 ) . `` A review of the mechanisms and
evidence for typical and atypical twinning '' . American Journal of Obstetrics and Gynecology. 214 ( 2 ) : 172 -- 91 . doi : 10.1016 / j. ajog. 2015.10. 930 .
PMID 26548710 . Jump up ^ Wenk , R.E.
</item 9>

<item 10>
Superfecundation
PMID 26548710 . Jump up ^ Wenk , R.E. ; Houtz , T ; Brooks , M ; Chiafari , F.A. ( 1992 ) . `` How frequent is heteropaternal superfecundation ? '' . Acta
geneticae medicae et gemellologiae. 41 ( 1 ) : 43 -- 7 . PMID 1488855 .
</item 10>
=== END ITEM id=00e27f466eed575f ===

=== ITEM id=d6aa8550032dd4e5 ===
Gold sentences and facts:
Sentence 0: 2016 production of still wine State Production ( gal ) Production ( % ) Alabama 34,966 0.004 % Arizona 190,008 0.024 % Arkansas 246,363 0.031 % California 680,272,512 84.354 % Colorado 556,994 0.069 % Connecticut 134,517 0.017 % Florida 1,634,103 0.203 % Georgia 276,144 0.034 % Idaho 497,007 0.062 % Illinois 391,805 0.049 % Indiana 1,411,540 0.175 % Iowa 335,522 0.042 % Kansas 104,129 0.013 % Kentucky 2,176,059 0.270 % Louisiana 38,232 0.005 % Maine 48,222 0.006 % Maryland 436,185 0.054 % Massachusetts 792,884 0.098 % Michigan 2,576,238 0.319 % Minnesota 331,946 0.041 % Missouri 993,831 0.123 % Montana 33,445 0.004 % Nebraska 120,366 0.015 % New Hampshire 159,316 0.020 % New Jersey 1,832,325 0.227 % New Mexico 749,818 0.093 % New York 27,969,308 3.468 % North Carolina 1,903,060 0.236 % Ohio 5,938,738 0.736 % Oklahoma 70,204 0.009 % Oregon 11,822,972 1.466 % Pennsylvania 12,405,181 1.538 % South Carolina 77,842 0.010 % South Dakota 139,738 0.017 % Tennessee 1,279,752 0.159 % Texas 1,907,299 0.237 % Vermont 2,172,526 0.269 % Virginia 2,157,395 0.268 % Washington 40,747,190 5.053 % West Virginia 40,733 0.005 % Wisconsin 1,129,405 0.140 % Others 312,051 0.039 % Sum 806,447,891 100 %
  Fact 0.0: The table reports the 2016 production of still wine.
  Fact 0.1: Alabama produced 34,966 gallons of still wine in 2016, which is 0.004 percent of production.
  Fact 0.2: Arizona produced 190,008 gallons of still wine in 2016, which is 0.024 percent of production.
  Fact 0.3: Arkansas produced 246,363 gallons of still wine in 2016, which is 0.031 percent of production.
  Fact 0.4: California produced 680,272,512 gallons of still wine in 2016, which is 84.354 percent of production.
  Fact 0.5: Colorado produced 556,994 gallons of still wine in 2016, which is 0.069 percent of production.
  Fact 0.6: Connecticut produced 134,517 gallons of still wine in 2016, which is 0.017 percent of production.
  Fact 0.7: Florida produced 1,634,103 gallons of still wine in 2016, which is 0.203 percent of production.
  Fact 0.8: Georgia produced 276,144 gallons of still wine in 2016, which is 0.034 percent of production.
  Fact 0.9: Idaho produced 497,007 gallons of still wine in 2016, which is 0.062 percent of production.
  Fact 0.10: Illinois produced 391,805 gallons of still wine in 2016, which is 0.049 percent of production.
  Fact 0.11: Indiana produced 1,411,540 gallons of still wine in 2016, which is 0.175 percent of production.
  Fact 0.12: Iowa produced 335,522 gallons of still wine in 2016, which is 0.042 percent of production.
  Fact 0.13: Kansas produced 104,129 gallons of still wine in 2016, which is 0.013 percent of production.
  Fact 0.14: Kentucky produced 2,176,059 gallons of still wine in 2016, which is 0.270 percent of production.
  Fact 0.15: Louisiana produced 38,232 gallons of still wine in 2016, which is 0.005 percent of production.
  Fact 0.16: Maine produced 48,222 gallons of still wine in 2016, which is 0.006 percent of production.
  Fact 0.17: Maryland produced 436,185 gallons of still wine in 2016, which is 0.054 percent of production.
  Fact 0.18: Massachusetts produced 792,884 gallons of still wine in 2016, which is 0.098 percent of production.
  Fact 0.19: Michigan produced 2,576,238 gallons of still wine in 2016, which is 0.319 percent of production.
  Fact 0.20: Minnesota produced 331,946 gallons of still wine in 2016, which is 0.041 percent of production.
  Fact 0.21: Missouri produced 993,831 gallons of still wine in 2016, which is 0.123 percent of production.
  Fact 0.22: Montana produced 33,445 gallons of still wine in 2016, which is 0.004 percent of production.
  Fact 0.23: Nebraska produced 120,366 gallons of still wine in 2016, which is 0.015 percent of production.
  Fact 0.24: New Hampshire produced 159,316 gallons of still wine in 2016, which is 0.020 percent of production.
  Fact 0.25: New Jersey produced 1,832,325 gallons of still wine in 2016, which is 0.227 percent of production.
  Fact 0.26: New Mexico produced 749,818 gallons of still wine in 2016, which is 0.093 percent of production.
  Fact 0.27: New York produced 27,969,308 gallons of still wine in 2016, which is 3.468 percent of production.
  Fact 0.28: North Carolina produced 1,903,060 gallons of still wine in 2016, which is 0.236 percent of production.
  Fact 0.29: Ohio produced 5,938,738 gallons of still wine in 2016, which is 0.736 percent of production.
  Fact 0.30: Oklahoma produced 70,204 gallons of still wine in 2016, which is 0.009 percent of production.
  Fact 0.31: Oregon produced 11,822,972 gallons of still wine in 2016, which is 1.466 percent of production.
  Fact 0.32: Pennsylvania produced 12,405,181 gallons of still wine in 2016, which is 1.538 percent of production.
  Fact 0.33: South Carolina produced 77,842 gallons of still wine in 2016, which is 0.010 percent of production.
  Fact 0.34: South Dakota produced 139,738 gallons of still wine in 2016, which is 0.017 percent of production.
  Fact 0.35: Tennessee produced 1,279,752 gallons of still wine in 2016, which is 0.159 percent of production.
  Fact 0.36: Texas produced 1,907,299 gallons of still wine in 2016, which is 0.237 percent of production.
  Fact 0.37: Vermont produced 2,172,526 gallons of still wine in 2016, which is 0.269 percent of production.
  Fact 0.38: Virginia produced 2,157,395 gallons of still wine in 2016, which is 0.268 percent of production.
  Fact 0.39: Washington produced 40,747,190 gallons of still wine in 2016, which is 5.053 percent of production.
  Fact 0.40: West Virginia produced 40,733 gallons of still wine in 2016, which is 0.005 percent of production.
  Fact 0.41: Wisconsin produced 1,129,405 gallons of still wine in 2016, which is 0.140 percent of production.
  Fact 0.42: Other states produced 312,051 gallons of still wine in 2016, which is 0.039 percent of production.
  Fact 0.43: The total 2016 production of still wine was 806,447,891 gallons, which is 100 percent.

Retrieved items:
<item 1>
American wine
American wine - wikipedia American wine Jump to : navigation , search American wine has been produced for over 300 years . Today , wine production is undertaken
in all fifty states , with California producing 89 percent of all US wine . The United States is the fourth - largest wine producing country in the world after
France , Italy , and Spain . The North American continent is home to several native species of grape , including Vitis labrusca , Vitis riparia , Vitis
rotundifolia , and Vitis vulpina . But the wine making industry is based on the cultivation of the European Vitis vinifera , which was introduced by European
settlers . With more than 1,100,000 acres ( 4,500 km ) under vine , the United States is the sixth-most planted country in the world after France , Italy ,
Spain , China and Turkey . Contents ( hide ) 1 History 2 Wine regions 2.1 Production by state 3 Appellation system 3.1 Appellation labeling laws 3.2
Semi-generic wines 3.3 Fighting varietals 4 Other U.S.
</item 1>

<item 2>
American wine has a 300-year history and is produced in all 50 states, though California accounts for about 84–89% of output; the U.S. ranks fourth globally.
European Vitis vinifera underpins the industry, alongside native and hybrid grapes. Early vineyards began in Florida, Virginia, Kentucky and California;
Prohibition devastated production, which later revived through university research and quality-focused winemaking. Major regions include the West Coast, New
York and other states. The
</item 2>

<item 3>
American wine
Rocky Mountain Region -- Notably Idaho and Colorado Southwestern United States -- Notably Texas and New Mexico Midwestern United States -- Notably Missouri and
Illinois Great Lakes region -- Notably Michigan , northern New York and Ohio East Coast of the United States -- Notably western New York State and eastern Long
Island , New Jersey , Pennsylvania , Virginia , North Carolina and Florida . Production by state ( edit ) Production of still wine per state in 2016 was as
follows : 2016 production of still wine State Production ( gal ) Production ( % ) Alabama 34,966 0.004 % Arizona 190,008 0.024 % Arkansas 246,363 0.031 %
California 680,272,512 84.354 % Colorado 556,994 0.069 % Connecticut 134,517 0.017 % Florida 1,634,103 0.203 % Georgia 276,144 0.034 % Idaho 497,007 0.062 %
Illinois 391,805 0.049 % Indiana 1,411,540 0.175 % Iowa 335,522 0.042 % Kansas 104,129 0.013 % Kentucky 2,
</item 3>

<item 4>
American wine
During the Great Depression , consumers demanded cheap `` jug wine '' ( so - called dago red ) and sweet , fortified ( high alcohol ) wine . Before Prohibition
dry table wines outsold sweet wines by three to one , but afterward , the ratio of demand changed dramatically . As a result , by 1935 , 81 % of California 's
production was sweet wines . For decades , wine production was low and limited . Leading the way to new methods of wine production was research conducted at the
University of California , Davis and some of the state universities in New York . Faculty at the universities published reports on which varieties of grapes
grew best in which regions , held seminars on winemaking techniques , consulted with grape growers and winemakers , offered academic degrees in viticulture ,
and promoted the production of quality wines . In the 1970s and 1980s , success by Californian winemakers in the northern part of the state helped to secure
foreign investment from other winemaking regions , most notably the Champenois of France . Winemakers also cultivated vineyards in Oregon and Washington , on
Long Island in New York , and numerous other new locales .
</item 4>

<item 5>
American wine
Although a Vitis vinifera variety , it is a grape of `` very modest '' quality . Jean - Louis Vignes was one of the early settlers to use a higher quality
vinifera in his vineyard near Los Angeles . The first winery in the United States to become commercially successful was founded in Cincinnati , Ohio in the
mid-1830s by Nicholas Longworth . He made a sparkling wine from Catawba grapes . By 1855 Ohio had 1500 acres in vineyards , according to travel writer Frederick
Law Olmsted , who said it was more than in Missouri and Illinois , which each had 1100 acres in wine . German immigrants from the late 1840s had been
instrumental in building the wine industry in those states . In the 1860s , vineyards in the Ohio River Valley were attacked by Black rot . This prompted
several winemakers to move north to the Finger Lakes region of western New York . During this time , the Missouri wine industry , centered on the German colony
in Hermann , was expanding rapidly along both shores of the Missouri River west of St. Louis .
</item 5>

<item 6>
American wine
labeling laws 5 Three - tier distribution 6 Largest producers 7 See also 8 References 9 Further reading 10 External links History ( edit ) Main article :
History of American wine Sparkling wine made from a second label of Pleasant Valley Wine Company in New York which was the first bonded winery in the United
States . The first Europeans to explore North America , a Viking expedition from Greenland , called it Vinland because of the profusion of grape vines they
found . The earliest wine made in what is now the United States was produced between 1562 and 1564 by French Huguenot settlers from Scuppernong grapes at a
settlement near Jacksonville , Florida . In the early American colonies of Virginia and the Carolinas , wine making was an official goal laid out in the
founding charters . However , settlers discovered that the wine made from the various native grapes had flavors which were unfamiliar and which they did not
like . This led to repeated efforts to grow the familiar European Vitis vinifera varieties , beginning with the Virginia Company exporting French vinifera vines
with French vignerons to Virginia in 1619 .
</item 6>

<item 7>
American wine
wine sales and is the second largest producer in the world . Constellation Brands - With foreign wine holdings Constellation is the largest producer in the
world and includes Robert Mondavi Winery and Columbia Winery in its portfolio The Wine Group - San Francisco - based business which owns the Franzia box wine
label , Concannon Vineyard and Mogen David kosher wine . Bronco Wine Company - Owners of the Charles Shaw wine `` Two Buck Chuck '' line which accounts for
nearly 5 million of Bronco 's annual average 9 million cases per year . Diageo - UK based company with American holdings in Sterling Vineyards , Beaulieu
Vineyard and Chalone Vineyard Brown - Forman Corporation - Owners of the Korbel Champagne Cellars brand Beringer Blass - Australian based wine division of
Foster 's Group and owner of the Beringer wine and Stags ' Leap Winery brands Jackson Wine Estates - Owners of the Kendall - Jackson brand See also ( edit )
Wine portal American Viticultural Area Globalization of wine Judgment of Paris References ( edit ) Jump up ^ United States Department of Agriculture `` Global
Wine Report August 2006 Archived April 8 , 2008 , at the Wayback Machine.
</item 7>

<item 8>
American wine
This Supreme Court decision meant that states could decide to allow both out - of - state wine sales and in - state sales , or ban both together . Convenience
stores and retail stores are large distributors of wine , with over 175,000 outlets that sell wine across the United States . In addition , there are around
332,000 other locations ( bars , restaurants , etc . ) that sell wine , contributing to the $30 + billion in annual sales over the past three years . In 2010 ,
the average monthly per - store sales of wine jumped to nearly $12,000 from $9,084 in 2009 . The average gross margin dollars from wine increased to $3,324 from
$2,616 in the year prior , with gross margin percentages up to an average 28.2 percent in 2010 , versus 27 percent in 2009 . Largest producers ( edit ) As of
2005 The largest producers of American wine . E & J Gallo Winery - Accounts for more than a quarter of all U.S. wine sales and is the second largest producer in
the world .
</item 8>

<item 9>
American wine
wine region and AVA portal All American Wineries U.S. winery and vineyard guide Free the Grapes : State wine shipment laws Doubtless as Good : Thomas Jefferson
's Dream for American Wine Fulfilled An online exhibition from the National Museum of American History , Smithsonian Institution American wine States Alabama
Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland
Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New Hampshire New Jersey New Mexico New York North Carolina North Dakota Ohio
Oklahoma Oregon Pennsylvania Rhode Island South Carolina South Dakota Tennessee Texas Utah Vermont Virginia Washington West Virginia Wisconsin Wyoming Wine by
country Africa Algeria Egypt Ethiopia Morocco Namibia South Africa Tanzania Tunisia Asia Armenia Azerbaijan Bhutan China Georgia India Iran Israel Japan Jordan
Kazakhstan Lebanon Nepal Palestine Philippines Vietnam Europe Albania Austria Belgium Bulgaria Croatia Cyprus Czech Republic Denmark France Germany Greece
Hungary Ireland Italy Latvia Liechtenstein Luxembourg Macedonia Malta Moldova Montenegro Netherlands Poland Portugal Romania Russia San Marino Serbia Slovakia
Slovenia Spain Sweden Switzerland Turkey Ukraine United Kingdom North America Canada Mexico United States Oceania Australia New Zealand South America Argentina
Bolivia Brazil Chile Paraguay Peru Uruguay Agriculture in the United States History History African - American history Industries Almond Apple Banana Blueberry
Cherry Christmas tree Citrus Corn Cotton Maple syrup Onion Pepper Rice Spinach Strawberry
</item 9>

<item 10>
American wine
Americans became more educated about wines and increased their demand for high - quality wine . All 50 states now have some acreage in vineyard cultivation . By
2004 , 668 million gallons ( 25.3 million hectoliters ) of wine were consumed in the United States . Today , the U.S. produces over 800 million gallons of wine
a year , of which California produces more than 84 % , followed by Washington , New York , Pennsylvania , and Oregon . In the second decade of the 21st century
, the US wine industry faces the growing challenges of competition from international exports and managing domestic regulations on interstate sales and shipment
of wine . Wine regions ( edit ) The Pacific Northwest includes in the winemaking regions of Washington State where varieties like Cabernet Sauvignon are grown .
There are nearly 3,000 commercial vineyards in the United States , and at least one winery in each of the 50 states . West Coast -- More than 90 % of the total
American wine production occurs in the states of California , Washington and Oregon .
</item 10>
=== END ITEM id=d6aa8550032dd4e5 ===

=== ITEM id=53d58cc82f7e6c23 ===
Gold sentences and facts:
Sentence 0: All black military pilots who trained in the United States trained at Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic .
  Fact 0.0: All black military pilots who trained in the United States trained at Moton Field.
  Fact 0.1: All black military pilots who trained in the United States trained at the Tuskegee Army Air Field.
  Fact 0.2: All black military pilots who trained in the United States were educated at Tuskegee University.
  Fact 0.3: Tuskegee University is located near Tuskegee, Alabama.
  Fact 0.4: The group included five Haitians from the Haitian Air Force.
  Fact 0.5: The group included one pilot from Trinidad.
  Fact 0.6: The group included a Hispanic or Latino airman born in the Dominican Republic.

Retrieved items:
<item 1>
Tuskegee Airmen
Census Bureau reported there were only 124 African - American pilots in the nation . The exclusionary policies failed dramatically when the Air Corps received
an abundance of applications from men who qualified , even under the restrictive requirements . Many of the applicants already had participated in the Civilian
Pilot Training Program , unveiled in late December 1938 ( CPTP ) . Tuskegee University had participated since 1939 . Testing ( edit ) This section does not cite
any sources . Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) (
Learn how and when to remove this template message ) The U.S. Army Air Corps had established the Psychological Research Unit 1 at Maxwell Army Air Field ,
Montgomery , Alabama , and other units around the country for aviation cadet training , which included the identification , selection , education , and training
of pilots , navigators , and bombardiers .
</item 1>

<item 2>
Tuskegee Airmen
Finally , on 3 April 1939 , Appropriations Bill Public Law 18 was passed by Congress containing an amendment by Senator Harry H. Schwartz , designating funds
for training African - American pilots . The War Department managed to put the money into funds of civilian flight schools willing to train black Americans .
War Department tradition and policy mandated the segregation of African - Americans into separate military units staffed by white officers , as had been done
previously with the 9th Cavalry , 10th Cavalry , 24th Infantry Regiment and 25th Infantry Regiment . When the appropriation of funds for aviation training
created opportunities for pilot cadets , their numbers diminished the rosters of these older units . In 1941 , the War Department and the Army Air Corps , under
pressure -- three months before its transformation into the USAAF -- constituted the first all - black flying unit , the 99th Pursuit Squadron . Due to the
restrictive nature of selection policies , the situation did not seem promising for African - Americans since , in 1940 , the U.S. Census Bureau reported there
were only 124 African - American pilots in the nation .
</item 2>

<item 3>
Tuskegee Airmen
This small number of enlisted men became the core of other black squadrons forming at Tuskegee and Maxwell Fields in Alabama . The Tuskegee program officially
began June 1941 with the 99th Pursuit Squadron at Tuskegee University . The unit consisted of 47 officers and 429 enlisted men , and was backed by an entire
service arm . After primary training at Moton Field , they were moved to the nearby Tuskegee Army Air Field , about 10 miles ( 16 km ) to the west for
conversion training onto operational types . Consequently , Tuskegee Army Air Field became the only Army installation performing three phases of pilot training
( basic , advanced , and transition ) at a single location . Initial planning called for 500 personnel in residence at a time . By mid-1942 , over six times
that many were stationed at Tuskegee , even though only two squadrons were training there . War poster featuring a Tuskegee Airman Tuskegee Army Airfield was
similar to already - existing airfields reserved for training white pilots , such as Maxwell Field , only 40 miles ( 64 km ) distant .
</item 3>

<item 4>
Graph query row:
entity: 99th Pursuit Squadron
relationship: LOCATED_IN
connected_to: Tuskegee University
located_in: Tuskegee University
</item 4>

<item 5>
Tuskegee Airmen
They formed the 332nd Fighter Group and the 477th Bombardment Group of the United States Army Air Forces . The name also applies to the navigators , bombardiers
, mechanics , instructors , crew chiefs , nurses , cooks and other support personnel . All black military pilots who trained in the United States trained at
Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from
the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic . Although the 477th
Bombardment Group trained with North American B - 25 Mitchell bombers , they never served in combat . The 99th Pursuit Squadron ( later , 99th Fighter Squadron
) was the first black flying squadron , and the first to deploy overseas ( to North Africa in April 1943 , and later to Sicily and Italy ) . The 332nd Fighter
Group , which originally included the 100th , 301st , and 302nd Fighter Squadrons , was the first black flying group .
</item 5>

<item 6>
Tuskegee Airmen
Some ground crews trained at Mather before rotating to Inglewood . Gunners learned to shoot at Eglin Field , Florida . Bombers - navigators learned their trades
at Hondo Army Air Field and Midland Air Field , Texas or at Roswell , New Mexico . Training of the new African - American crewmen also took place at Sioux Falls
, South Dakota , Lincoln , Nebraska , and Scott Field , Belleville , Illinois . Once trained , the air and ground crews would be spliced into a working unit at
Selfridge . Command difficulties ( edit ) The new group 's first commanding officer was Colonel Robert Selway , who had also commanded the 332nd Fighter Group
before it deployed for combat overseas . Like his ranking officer , Major General Frank O'Driscoll Hunter from Georgia , Selway was a racial segregationist .
Hunter was blunt about it , saying such things as , `` ... racial friction will occur if colored and white pilots are trained together . '' He backed Selway 's
violations of Army Regulation 210 - 10 , which forbade segregation of air base facilities .
</item 6>

<item 7>
Tuskegee Airmen
Psychologists employed in these research studies and training programs used some of the first standardized tests to quantify IQ , dexterity and leadership
qualities to select and train the best - suited personnel for the roles of bombardier , navigator , and pilot . The Air Corps determined that the existing
programs would be used for all units , including all - black units . At Tuskegee , this effort continued with the selection and training of the Tuskegee Airmen
. The War Department set up a system to accept only those with a level of flight experience or higher education which ensured that only the most able and
intelligent African - American applicants were able to join . The first Lady 's flight ( edit ) The budding flight program at Tuskegee received a publicity
boost when First Lady Eleanor Roosevelt inspected it in March 1941 , and flew with African - American chief civilian instructor C. Alfred `` Chief '' Anderson .
Anderson , who had been flying since 1929 , and was responsible for training thousands of rookie pilots , took his prestigious passenger on a half - hour flight
in a Piper J - 3 Cub . After landing , she cheerfully announced , `` Well , you can fly all right . ''
</item 7>

<item 8>
Tuskegee Airmen
The white population of Freeman Field was 250 officers and 600 enlisted men . Superimposed on it were 400 African - American officers and 2,500 enlisted men of
the 477th and its associated units . Freeman Field had a firing range , usable runways , and other amenities useful for training . African - American airmen
would work in proximity with white ones ; both would live in a public housing project adjacent to the base . Colonel Selway turned the non-commissioned officers
out of their club and turned it into a second officers club . He then classified all white personnel as cadre , and all African - Americans as trainees . One
officers club became the cadre 's club . The old Non-Commissioned Officers Club , promptly sarcastically dubbed `` Uncle Tom 's Cabin '' , became the trainee 's
officers club . At least four of the trainees had flown combat in Europe as fighter pilots , and had about four years in service . Four others had completed
training as pilots , bombardiers and navigators , and may have been the only triply qualified officers in the entire Air Corps .
</item 8>

<item 9>
Tuskegee Airmen
This was one of the earliest racially integrated courses in the U.S. Army . Seventeen flight surgeons served with the Tuskegee Airmen from 1941 - 49 . At that
time , the typical tour of duty for a U.S. Army flight surgeon was four years . Six of these physicians lived under field conditions during operations in North
Africa , Sicily , and Italy . The chief flight surgeon to the Tuskegee Airmen was Dr. Vance H. Marchbanks , Jr. , MD , a childhood friend of Benjamin Davis .
The accumulation of washed - out cadets at Tuskegee and the propensity of other commands to `` dump '' African - American personnel on the post exacerbated the
difficulties of administering Tuskegee . A shortage of jobs for them made these enlisted men a drag on Tuskegee 's housing and culinary departments . Trained
officers were also left idle , as the plan to shift African - American officers into command slots stalled , and white officers not only continued to hold
command , but were joined by additional white officers assigned to the post .
</item 9>

<item 10>
Tuskegee Airmen
Several of the Tuskegee Airmen had logged over 900 flight hours by this time . Nevertheless , by Colonel Selway 's fiat , they were trainees . Off - base was no
better ; many businesses in Seymour would not serve African - Americans . A local laundry would not wash their clothes , yet willingly laundered those of
captured German soldiers . In early April 1945 , the 118th Base Unit transferred in from Godman Field ; its African - American personnel held orders that
specified they were base cadre , not trainees . On 5 April , officers of the 477th peaceably tried to enter the whites - only Officer 's Club . Selway had been
tipped off by a phone call , and had the assistant provost marshal and base billeting manager stationed at the door to refuse the 477th officers entry . The
latter , a major , ordered them to leave , and took their names as a means of arresting them when they refused . It was the beginning of the Freeman Field
Mutiny .
</item 10>
=== END ITEM id=53d58cc82f7e6c23 ===

=== ITEM id=ec729d7a1b188d01 ===
Gold sentences and facts:
Sentence 0: In episode 3 , `` Hugs Can Be Deceiving '' , Suzanne 's backstory is revealed via a flashback . She was adopted by white parents and raised in the suburbs , but her mental instability alienated her classmates and their parents alike . Her adoptive mother pushed her to excel at everything she did to prove that she was as good as everyone else ; ironically , the constant pressure only made Suzanne 's psychological problems worse . In the same episode , it is revealed that Suzanne had run outside and knocked Chapman unconscious after Chapman had beaten Tiffany `` Pennsatucky '' Doggett ( Taryn Manning ) unconscious at the end of season 1 .
  Fact 0.0: In episode 3, titled 'Hugs Can Be Deceiving', Suzanne's backstory is revealed via a flashback.
  Fact 0.1: Suzanne was adopted by white parents.
  Fact 0.2: Suzanne was raised in the suburbs.
  Fact 0.3: Suzanne's mental instability alienated her classmates.
  Fact 0.4: Suzanne's mental instability alienated her classmates' parents.
  Fact 0.5: Suzanne's adoptive mother pushed her to excel at everything she did, to prove she was as good as everyone else.
  Fact 0.6: Ironically, the constant pressure from Suzanne's adoptive mother made Suzanne's psychological problems worse.
  Fact 0.7: In the same episode, it is revealed that Suzanne had run outside.
  Fact 0.8: In the same episode, Suzanne knocked Chapman unconscious.
  Fact 0.9: Suzanne knocked Chapman unconscious after Chapman had beaten Tiffany 'Pennsatucky' Doggett (Taryn Manning) unconscious.
  Fact 0.10: Chapman beat Doggett unconscious at the end of season 1.

Retrieved items:
<item 1>
Crazy Eyes (character)
In the season finale , Suzanne assists Poussey and Taystee in caring for Brook Soso , whom they discover unconscious from a drug overdose , and develops a
closer bond with Maureen in the final scene , owing to a misunderstanding over a turtle that bites Maureen 's foot . Season 4 ( edit ) In season 4 it is
revealed that Suzanne 's crime was kidnapping and accidental manslaughter of a child she grew friendly with while working as a store greeter . Believing that
she and the boy were simply being friends , she did not realize the implications of her actions when she came across him playing in the park and invited him to
her apartment for ice cream and video games . The boy went willingly , but soon became frightened and wanted to call his dad . While Suzanne is out of the room
preparing for what she plans as more fun , he calls 911 . Suzanne becomes agitated as she tells him that 911 should only be called when there is an emergency .
The child panics and tries to get away . He goes onto the fire escape and falls to his death . It is mentioned that she took a plea deal .
</item 1>

<item 2>
Crazy Eyes (character)
Suzanne gives a statement to prison officials confessing to the attack , but inmate counselor Sam Healy ( Michael J. Harney ) produces evidence that exonerates
her . When she realizes that Vee used her , she breaks into tears . In episode 3 , `` Hugs Can Be Deceiving '' , Suzanne 's backstory is revealed via a
flashback . She was adopted by white parents and raised in the suburbs , but her mental instability alienated her classmates and their parents alike . Her
adoptive mother pushed her to excel at everything she did to prove that she was as good as everyone else ; ironically , the constant pressure only made Suzanne
's psychological problems worse . In the same episode , it is revealed that Suzanne had run outside and knocked Chapman unconscious after Chapman had beaten
Tiffany `` Pennsatucky '' Doggett ( Taryn Manning ) unconscious at the end of season 1 . Season 3 ( edit ) In season 3 , Taystee has her hands full trying to
control Suzanne , who is still grieving Vee 's death and is unstable , prone to losing her temper and constantly fighting with Poussey .
</item 2>

<item 3>
Crazy Eyes (character)
It is mentioned that she took a plea deal . When the inmates take advantage of a temporary lack of security and enjoy the nearby lake and woods , Kukudio
insists that she and Suzanne run away together . Suzanne refuses and their relationship becomes rocky . In `` People Persons , CO Humphrey commands the two of
them to fight . Suzanne rejects this but Maureen , who wants to fight , provokes her by launching into a stream of insults . Suzanne fully retaliates by
throwing Maureen to the ground and beating her repeatedly , until the Dominicans pull her off . This fight leads her to extreme guilt and a wish to punish
herself . She demonstrates this especially when a peaceful protest in the jail cafeteria turns into a chaotic scene and Poussey Washington dies by suffocation .
Whilst the inmates grieve over her death , Suzanne believes that Poussey is alive and tries numerous times to suffocate herself . Finally , she is sent to
Medical and is placed in a bed next to Kukudio . Season 5 ( edit ) In season five , Suzanne continues to recover from her injuries and also goes out of her way
to help Maureen , whose injuries are severe .
</item 3>

<item 4>
Crazy Eyes (character)
September 17 , 2017 . Retrieved September 20 , 2017 . ( hide ) Orange Is the New Black Characters Piper Chapman Alex Vause Crazy Eyes Nicky Nichols Tiffany
Doggett Lorna Morello Miss Rosa Poussey Washington Seasons `` Lesbian Request Denied '' 5 Related Piper Kerman Orange Is the New Black : My Year in a Women 's
Prison `` You 've Got Time '' Retrieved from `` https://en.wikipedia.org/w/index.php?title=Crazy_Eyes_(character)&oldid=814872296 '' Categories : Orange Is the
New Black characters Fictional adoptees Fictional African - American people Fictional characters introduced in 2013 Fictional criminals Fictional lesbians
Fictional prison escapees Fictional writers Fictional LGBT characters in television Fictional inmates in American federal prisons Hidden categories : Use mdy
dates from September 2016 Articles needing additional references from June 2017 All articles needing additional references Talk Contents About Wikipedia Türkçe
Edit links This page was last edited on 11 December 2017 , at 12 : 19 . About Wikipedia
</item 4>

<item 5>
Crazy Eyes (character)
Suzanne nicknames Chapman `` Dandelion '' , because of her blonde hair and puts in a request to become Chapman 's bunkmate . Nonetheless , Chapman resists her
advances . As a result , Suzanne urinates on the floor in Chapman 's bunk quarters . In episode 9 , `` Fucksgiving '' , her white adoptive parents , Pat and
Dennis Warren , are introduced . Season 2 ( edit ) As she rebounds from her unrequited love for Chapman , Suzanne is befriended by veteran inmate Yvonne `` Vee
'' Parker ( Lorraine Toussaint ) as Vee develops her powerbase in the prison . Vee takes advantage of Suzanne 's desperation for affection , and Suzanne becomes
Vee 's fanatically loyal right - hand woman . Suzanne serves Vee as muscle , even beating up her friend Poussey Washington ( Samira Wiley ) on Vee 's orders .
When Vee attacks prison matriarch Galina `` Red '' Reznikov ( Kate Mulgrew ) and puts her in the hospital ward , she manipulates Suzanne into believing that she
committed the crime .
</item 5>

<item 6>
Crazy Eyes (character)
Crazy Eyes ( character ) - wikipedia Crazy Eyes ( character ) Jump to : navigation , search Suzanne `` Crazy Eyes '' Warren Orange Is the New Black character
Uzo Aduba as Suzanne `` Crazy Eyes '' Warren in a promotional poster for season 2 of Orange Is the New Black First appearance `` I Was n't Ready '' Season 1 ,
episode 1 July 11 , 2013 Portrayed by Uzo Aduba Eden Wiggins ( at age 10 ) Taliyah Whitaker ( at age 5 ) Information Significant other ( s ) Maureen Kukudio (
ex-girlfriend ) Parents Pat and Dennis Warren Suzanne `` Crazy Eyes '' Warren is a fictional character played by Uzo Aduba on the Netflix series Orange Is the
New Black . Warren is portrayed as intelligent , but lacking in social skills , and prone to spiral into emotional outbursts when agitated . The character is
the only role that has received Emmy Award recognition both in the comedy and drama genres from the same show and only the second character to earn Emmy
recognition in both genres .
</item 6>

<item 7>
Crazy Eyes (character)
`` Orange Is the New Black Reveals Crazy Eyes ' Tragic Crime '' . ENews . Retrieved 18 September 2017 . Jump up ^ Meltzer , Tom ( December 17 , 2013 ) . `` The
best TV of 2013 : No 6 -- Orange is the New Black ( Netflix ) : The US prison drama -- Netflix 's most - watched show of the year -- is a binge - watcher 's
dream : moreish and full of surprises '' . The Guardian . Retrieved October 5 , 2014 . Jump up ^ Jancelewicz , Chris ( July 3 , 2014 ) . `` Orange Is The New
Black Review : Time Behind Bars Should n't Be This Fun '' . The Huffington Post Canada . Retrieved October 6 , 2014 . Jump up ^ `` Primetime Emmy Awards 2014 :
The winners list '' . CNN . August 26 , 2014 . Retrieved September 4 , 2014 . Jump up ^ `` The International Press Academy Announces Nominations For The 18th
Annual Satellite Awards '' . PR Newswire .
</item 7>

<item 8>
Crazy Eyes (character)
Jump up ^ Schaefer , Megan ( June 3 , 2014 ) . `` ' Orange Is The New Black ' Season 1 Recap : Everything To Know Before The Season 2 Premiere '' .
International Business Times . Retrieved September 6 , 2014 . Jump up ^ Foster , Lewis ( June 19 , 2014 ) . `` Visual Summary : Orange Is The New Black Season 1
'' . Paste . Retrieved September 6 , 2014 . Jump up ^ O'Sullivan , Erin ( August 14 , 2013 ) . `` Orange Is The New Black : Uzo Aduba ( Suzanne Warren )
Promoted To Series Regular '' . Access Hollywood . Retrieved September 6 , 2014 . Jump up ^ Henderson , Danielle ( August 16 , 2013 ) . `` Orange Is the New
Black Episode Nine Recap : We Finally Experience the SHU '' . Vulture . Retrieved September 6 , 2014 . Jump up ^ `` Orange is the New Black Season 1 Episode 9 :
F * cksgiving : Episode Cast & Crew '' . TV.com .
</item 8>

<item 9>
Crazy Eyes (character)
Deadline.com . Retrieved September 5 , 2014 . Jump up ^ Kohan , Jenji ( writer / creator ) ; Herrmann , Tara ( producer ) ; Burley , Mark ( producer ) ( May 13
, 2014 ) . Orange Is the New Black : Season 1 -- `` I Was n't Ready '' audio commentary ( Blu - ray ) . Lionsgate . Event occurs at 37 : 00 -- 37 : 30 . Jump up
^ `` Uzo Aduba '' . Internet Movie Database . Retrieved June 12 , 2015 . Jump up ^ `` Orange is the New Black Season 2 Episode 3 : Hugs Can Be Deceiving '' .
TV.com . Retrieved September 5 , 2014 . Jump up ^ Potts , Kimberly ( June 18 , 2014 ) . `` The 10 Best ' Orange Is the New Black ' Character Backstories '' .
Yahoo ! TV . Retrieved September 5 , 2014 . Jump up ^ White , Chelsea F.C. ( October 2 , 2013 ) . `` Spot the difference !
</item 9>

<item 10>
Crazy Eyes (character)
Eventually , she turns to new prison counsellor Berdie Rogers 's drama class . There , she begins writing a surrealistic science - fiction erotica series called
`` The Time Hump Chronicles '' which , although considered obscene and strange by both Rogers and Taystee , becomes an instant hit amongst the inmates .
Although they prove to be therapeutic for her mental health , Suzanne becomes stressed and irritated with the constant harassment by the other inmates for more
material , the persistent ideas that are being floated past her and the fan - fiction that is left at her bunk or lying around . Soon , some of the extracts
find their way into the hands of the COs , resulting in Officer Donaldson being mocked by both his colleagues and the inmates alike upon the realization that he
is the inspiration for one of the characters . This leads to Rogers ' suspension . Meanwhile , Suzanne is taken by surprise when she discovers that one of her
more prolific fans , Maureen Kukudio , is actually interested in her romantically . Suzanne seeks advice from Lorna about how to respond ; she admits she finds
Maureen attractive , but she has never had a girlfriend before and subsequently feels she must back away from a sexual liaison .
</item 10>
=== END ITEM id=ec729d7a1b188d01 ===

=== ITEM id=557aeb4dea25a320 ===
Gold sentences and facts:
Sentence 0: In most of the world speed limit signs display the limitation within a red circle . This design follows the style set out by the Vienna Convention on Road Signs and Signals , with the exception of United States and Canada . All countries globally worldwide ( except the United Kingdom and United States ) use metric speed limits in kilometres per hour . Metric signs are optional in the United States though less commonly seen , while Samoa uses both units . In Ontario , Canada , the type , location , and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act .
  Fact 0.0: In most of the world, speed limit signs display the limitation within a red circle.
  Fact 0.1: The red-circle speed limit sign design follows the style set out by the Vienna Convention on Road Signs and Signals.
  Fact 0.2: The Vienna Convention style is not followed by the United States and Canada.
  Fact 0.3: All countries worldwide, except the United Kingdom and the United States, use metric speed limits in kilometres per hour.
  Fact 0.4: Metric speed limit signs are optional in the United States.
  Fact 0.5: Metric speed limit signs are less commonly seen in the United States.
  Fact 0.6: Samoa uses both units for speed limits.
  Fact 0.7: In Ontario, Canada, the type, location, and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act.

Retrieved items:
<item 1>
Speed limit
In international European roads , speed should be taken into account at design stage . Maximum speed recommended by international European road design standard
Road classification 60 km / h 80 km / h ( 50 mph ) 100 km / h ( 60 mph ) 120 km / h ( 75 mph ) 140 km / h ( 85 mph ) Motorway x 80 100 120 140 Express road 60
80 100 120 x Road 60 80 100 x x Minimum speed limits ( edit ) UK minimum speed limit sign Some roads also have `` minimum speed limits '' , where slow speeds
can impede traffic flow or be dangerous . Signs often use blue circles based on the obligatory signs of the Vienna Convention on Road Signs and Signals . A
Japanese minimum speed sign has the same design as a maximum speed sign , but with a horizontal line below the number . In the United States of America , they
are also identical to their respective maximum speed limit signs with the exception of the text `` MINIMUM SPEED '' . This minimum speed is not so common , as
the risk usually raise with higher speed and not with lower speed .
</item 1>

<item 2>
Speed limit
In the European Union , large signposts showing the national ( default ) speed limits of the respective country are usually erected immediately after border
crossings , with a repeater sign some 200 to 500 m ( 660 to 1,640 ft ) after the first sign . Some places provide an additional `` speed zone ahead '' ahead of
the restriction and speed limit reminder signs may appear at regular intervals which may be painted on the road surface . Signs are normally placed on both
sides of the road and in some places there are small ( less than 1 / 4 the size of the sign ) rectangular orange reflector flags attached to both upper right
corners of both signs . The speed limit sign marking the new speed zone may also have the orange flags ; this practice can be observed in New York on highways
where the speed limit varies such as New York State Route 17 . Speed limits on European Motorways in km / h Signage in many countries , especially in Europe ,
conforms to the Vienna Convention on Road Signs and Signals using black text with a red circle on a white background . In the United States , the signs are
usually rectangular with the words `` SPEED LIMIT '' and the values in black on a white background .
</item 2>

<item 3>
Speed limit
p. 343 . External links ( edit ) Wikimedia Commons has media related to Speed limit signs and Driver speed and relative risk of an accident . Global map of
speed limits from Wikispeedia A Walk Through The History of Speed Limit Signs in The US John F. Carr 's State Traffic and Speed Laws Global map of speed limits
from OpenStreetMap Traffic signs Signs Warning sign Regulatory sign Prohibitory traffic sign Mandatory sign Special regulation sign Stop sign Yield sign
Crossbuck Dead End sign One - way traffic Speed limit ( by country ) Advisory speed limit Information sign Direction , position , or indication sign Variable -
message sign Bilingual sign Driver location signs Logo sign Priority to the right By comparison Comparison of European road signs Comparison of MUTCD -
influenced traffic signs Comparison of traffic signs in English - speaking countries By country Africa Botswana Mauritius South Africa Zimbabwe Asia Azerbaijan
Bangladesh Brunei Cambodia China Hong Kong India Indonesia Iran Israel Japan Laos Malaysia Nepal Pakistan Singapore South Korea Sri Lanka Thailand Turkey United
Arab Emirates Vietnam Europe Austria Bosnia and Herzegovina Czech Republic Denmark Estonia Finland France Germany Greece Iceland Ireland Italy Liechtenstein
Netherlands Norway Poland Sweden Switzerland Ukraine United Kingdom Scotland Wales North America Canada Mexico United States Oceania Australia New Zealand South
America Argentina Brazil Chile Colombia Lights Traffic
</item 3>

<item 4>
Speed limit
The U.S. Federal Highway Administration 's Manual on Uniform Traffic Control Devices provides guidelines for the appearance of speed limit signs . In Alaska and
California , speed limits are often labeled `` MAXIMUM SPEED '' instead . In Oregon , most speed limit signs just say `` SPEED '' and the number . Canada has
similar signs bearing the legend `` MAXIMUM '' and in km / h instead of mph . `` Maximum '' is used instead of `` Speed Limit '' because it has similar meanings
in English and French . Australian signs are rectangular but have a red circle like the Vienna Convention signs . The speed limit is commonly set at or below
the 85th percentile operating speed ( being the speed which no more than 15 % of traffic is exceeding ) and in the US is frequently set 4 to 8 mph ( 6 to 13 km
/ h ) below that speed . Thus , if the 85th percentile operating speed as measured by a Traffic and Engineering Survey exceeds the design speed , legal
protection is given to motorists traveling at such speeds ( design speed is `` based on conservative assumptions about driver , vehicle and roadway
characteristics '' ) .
</item 4>

<item 5>
Speed limit
Per distance traveled , consequences of inappropriate speed are more frequent on lower speed , lower quality roads ; in the United States , for example , the ``
speeding fatality rate for local roads is three times that for Interstates '' Citations for violations of the basic speed law without a crash have sometimes
been ruled unfairly vague or arbitrary , hence a violation the due process of law , at least in the State of Montana . Even within states , differing
jurisdictions ( counties and cities ) choose to prosecute similar cases with differing approaches . Maximum speed limits ( edit ) A speed limit sign in Ontario
In France city sign has value of speed limit A street limited at 30 km / h End of speed limit , previous speed limit applies Most public roads in most countries
have a legally assigned numerical maximum speed limit which applies on all roads unless otherwise stated ; lower speed limits are often shown on a sign at the
start of the restricted section , although the presence of streetlights or the physical arrangement of the road may sometimes also be used instead . A posted
speed limit may only apply to that road or to all roads beyond the sign that defines them depending on local laws .
</item 5>

<item 6>
Speed limit
Speed limit - wikipedia Speed limit Jump to : navigation , search This article is about road speed limits . For speed of light , see special relativity . For
road speed limits in specific countries , see Speed limits by country . For rail speed limits , see Slow zone . A typical 60 km / h speed limit sign used in
Australia A typical 50 mph speed limit sign in the United States A typical 50 mph speed limit sign in the United Kingdom and 50 km / h speed limit sign in Hong
Kong Road speed limits are used in most countries to set the maximum ( or minimum in some cases ) speed at which road vehicles may legally travel on particular
stretches of road . Speed limits may be variable and in some places speed is unlimited ( e.g. on some Autobahn sections in Germany ) . Speed limits are normally
indicated on a traffic sign . Speed limits are commonly set by the legislative bodies of nations or provincial governments and enforced by national or regional
police or judicial bodies . The first maximum speed limit was the 10 mph ( 16 km / h ) limit introduced in the United Kingdom in 1861 .
</item 6>

<item 7>
Speed limit
1992 . Retrieved 2010 - 07 - 10 . fatal accident rates on rural Interstate highways posted at 65 mph or rural non-Interstate highways posted at 55 mph had not
significantly changed after the implementation of the 65 - mph speed limit . Jump up ^ `` South Dakota Speed Monitoring '' ( PDF ) . South Dakota Department of
Transportation . 2012 . Retrieved 2014 - 12 - 19 . Jump up ^ `` Written Answers to Questions : Road Accidents '' . Hansard . House of Commons . 2003 - 01 - 31 .
Retrieved 2010 - 04 - 23 . TRL research on urban speed management methods published in 1998 ( TRL Report 363 ) found only an average 1 mph drop in speeds and no
discernible accident reduction in accidents in 20 mph limits using only signs . Advisory speed limits are not normally approved in England and Wales . However ,
the more successful 20 mph zones that use self enforcing traffic calming features achieved average speed reductions of around 10 mph which produced a 70 per
cent .
</item 7>

<item 8>
Speed limit
189 , p. 335 , ( California Supreme Court reporter ) . July 27 , 1922 . Retrieved 2013 - 07 - 27 . This is but a reiteration of the rule , in statutory form ,
which has always been in force without regard to a statutory promulgation to the effect that drivers or operators of vehicles , and more particularly motor
vehicles , must be specially watchful in anticipation of the presence of others at places where other vehicles are constantly passing , and where men , women ,
and children are liable to be crossing , such as corners at the intersections of streets or other similar places or situations where people are likely to fail
to observe an approaching automobile . ^ Jump up to : www.destatis.de ( 10 July 2013 ) . `` Unfallentwicklung auf deutschen Straßen 2012 ( Crashes on German
Roads 2012 ) '' ( PDF ) . Statistisches Bundesamt ( Federal Statistics Office ) . Statistisches Bundesamt . Retrieved 2013 - 09 - 23 .
</item 8>

<item 9>
Speed limit
Variable speed limits ( edit ) Example variable speed limit sign in the United States Digital speed limit sign for variable speed limits In 1965 , the first
known experiments with variable speed limit signs took place on a 30 km stretch of German motorway A8 between Munich and the border city of Salzburg , Austria .
Mechanically variable message signs could display speeds of 60 , 80 and 100 km / h , `` danger zone '' or `` accident '' . Personnel monitored traffic using
video technology , and manually controlled the signage . Beginning in the 1970s , additional advanced traffic control systems were put into service . Modern
motorway control systems can work without human intervention using various types of sensors to measure traffic flow and weather conditions . In 2009 , 1,300 km
( 810 mi ) of German motorways were equipped with such systems . In the late 1960s , heavily traveled portions of the New Jersey Turnpike began using variable
speed limit signs , in combination with variable message signs . Officials can adjust the speed limit according to weather , traffic conditions , and
construction .
</item 9>

<item 10>
Speed limit
The speeding fatality rate for local roads is three times that for Interstates Jump up ^ http://www.pe.com/articles/speed-776756-survey-limit.html Jump up ^
http://caselaw.findlaw.com/mt-supreme-court/1110919.html Jump up ^
http://vincenzeslaw.com/4-ways-virginia-reckless-driving-charges-handled-differently-fairfax-county-arlington-alexandria-stafford/ Jump up ^ `` Traffic
Operations : Road Signs '' . Government of South Australia . 2002 - 08 - 23 ( last update date ) . Retrieved 2010 - 09 - 26 . Most regulatory signs are
rectangular with a black legend on a white background ... ( exceptions include ) speed restriction signs ( symbol enclosed in a red circle ) '' Check date
values in : date = ( help ) Jump up ^ `` Speed Zoning Information '' ( PDF ) . Institute of Transportation Engineers . 22 March 2004 . Retrieved 2009 - 09 - 17
. Jump up ^ MUTCD Sections 2B.
</item 10>
=== END ITEM id=557aeb4dea25a320 ===
