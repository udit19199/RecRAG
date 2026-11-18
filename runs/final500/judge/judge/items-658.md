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

This file has 7 items.

=== ITEM id=fb54cd722d5f925e ===
Gold sentences and facts:
Sentence 0: `` Bad Boys '' is a 1987 song by the Jamaican reggae band Inner Circle , which obtained high popularity in the United States after its re-release in 1993 , peaking at number 8 on the Billboard Hot 100 and number 7 on the Top 40 Mainstream . It is well known as the opening theme to the U.S. TV show Cops .
  Fact 0.0: Bad Boys is a 1987 song.
  Fact 0.1: Bad Boys is a song by the Jamaican reggae band Inner Circle.
  Fact 0.2: Bad Boys obtained high popularity in the United States after its re-release in 1993.
  Fact 0.3: Bad Boys peaked at number 8 on the Billboard Hot 100.
  Fact 0.4: Bad Boys peaked at number 7 on the Top 40 Mainstream.
  Fact 0.5: Bad Boys is well known as the opening theme to the U.S. TV show Cops.

Retrieved items:
<item 1>
NBC Sunday Night Football
A new Underwood - sung theme known as `` Oh , Sunday Night '' , which takes elements from her 2014 duet hit with Miranda Lambert , `` Somethin ' Bad '' ,
premiered with the opening game of the 2016 season on September 11 . For NBC 's coverage of Thursday Night Football , a capella group Pentatonix sings the
opening theme song called `` Weekend Go . '' The song is a reworked version of their original song `` Sing . '' NBC also used the `` TNF '' main theme music
called `` Ca n't Hold Us Down , '' performed by members of the Broadway orchestra for Hamilton . Show opening ( edit ) The song is at the centerpiece of the
opening montage , which has changed several times over the years . Underwood 's music has always played in the background over the official welcome after the
opening is completed and the teams take the field . 2006 ( edit ) For the first season , Pink appeared to sing from the top of a skyscraper as a helicopter
zoomed down on a city skyline with enlarged players Shaun Alexander , LaDainian Tomlinson and Tom Brady and the field , the results of computer - generated
imagery .
</item 1>

<item 2>
Bad Boys (Inner Circle song)
It is well known as the opening theme to the U.S. TV show Cops . Contents ( hide ) 1 History 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 In
popular culture 4 See also 5 References 6 External links History ( edit ) The song was originally released in 1987 on the album One Way and obtained minor
airplay and poor sales . However , in 1992 , it was also included on the Bad to the Bone album , and the song was released as a single in 1993 , after the
unexpected success of their previous international hit single `` Sweat ( A La La La La Long ) '' . `` Bad Boys '' reached number 52 in the United Kingdom and
number 8 in the United States ; in the latter territory , it was certified Gold and sold 600,000 copies . The song was also released as B - side on Inner Circle
's 1992 single `` Sweat ( A La La La La Long ) '' . The song was also released as a single in mid-1991 in Norway ; it peaked at number one for three weeks .
</item 2>

<item 3>
NBC Sunday Night Football
In the South Park episode entitled `` Faith Hilling '' , Eric Cartman sings an obvious spoof of Hill 's actual Sunday Night Football song . On April 15 , 2013 ,
Hill announced that she would no longer sing the intro song for Sunday Night Football . The use of the reworked Joan Jett song is another similarity to ABC 's
Monday Night Football coverage . From 1989 to 2011 and again beginning in 2017 , Hank Williams , Jr. opens MNF with a reworking of his song `` All My Rowdy
Friends Are Coming Over Tonight '' entitled `` All My Rowdy Friends Are Back for Monday Night ( Are You Ready for Some Football ? ) '' . On October 7 , 2012 ,
the Soul Rebels Brass Band had a featured performance on Sunday Night Football , performing the show 's theme song . It was announced May 7 , 2013 , that Carrie
Underwood would take over singing the theme song .
</item 3>

<item 4>
NBC Sunday Night Football
Initially , the stars presented were Aaron Rodgers ( Green Bay Packers ) , Clay Matthews ( Packers ) , Ray Lewis ( Baltimore Ravens ) , Patrick Willis ( San
Francisco 49ers ) , Jimmy Graham ( New Orleans Saints ) , DeMarcus Ware ( Dallas Cowboys ) , Calvin Johnson ( Detroit Lions ) , Larry Fitzgerald ( Arizona
Cardinals ) , Jared Allen ( Minnesota Vikings ) , Rob Gronkowski ( New England Patriots ) , Darrelle Revis ( New York Jets ) and Eli Manning ( New York Giants )
. The opening itself was remixed with Nick Mangold ( Jets ) and LeSean McCoy ( Philadelphia Eagles ) filling in for Ray Lewis and Darrelle Revis . Beginning
with this season , the opening title sequence was not used at the start of the NBC Sunday Night Football Thanksgiving Special broadcast . This opening was also
not used on December 16 , 2012 , two days after the Sandy Hook shooting . 2013 ( edit ) Carrie Underwood became the performer for the theme song for the 2013
season , replacing Faith Hill . Her intro debuted on September 8 , with the theme arrangement itself tilted even more towards country to reflect the change in
singers .
</item 4>

<item 5>
Let My Love Open the Door
Songfacts.com . Jump up ^ `` Pete Townshend UK charts '' . Official Charts . Jump up ^ `` Pete Townshend US charts '' . Billboard . Jump up ^ Joel Whitburn 's
Top Pop Singles 1955 - 1990 - ISBN 0 - 89820 - 089 - X Jump up ^ Collectionscanada.gc.ca Archived 2016 - 04 - 25 at the Wayback Machine . Jump up ^
Musicoutfitters.com Jump up ^ `` The Covers EP : Various artists : MP3 Downloads '' . Amazon.com . Retrieved 2016 - 10 - 16 . Jump up ^ `` Help us open the door
to better cancer care # LetMyLoveOpenTheDoor '' . Firstcitizens.com . Retrieved 2016 - 10 - 16 . Jump up ^ `` Walmart TV Commercial , ' Christmas Like a Rock
Star ' Song by Pete Townshend '' . Ispot.tv . Retrieved December 6 , 2017 . Jump up ^ `` Leathal Weapon ( TV series ) `` Fools Rush '' Soundtracks ( 2017 ) `` .
</item 5>

<item 6>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 6>

<item 7>
Law & Order: Special Victims Unit
Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda
Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13
14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' `` Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice
Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion '' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion
'' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and nominations Law & Order franchise Chicago franchise Chicago Fire
Characters Matthew Casey Kelly Severide Episodes Season 1 5 6 Chicago P.D.
</item 7>

<item 8>
Stranger Things
The show 's theme is based on an unused work Stein composed much earlier that ended up in the library of work they shared with the production staff , who
thought that with some reworking would be good for the opening credits . In addition to original music , Stranger Things features period music from artists
including The Clash , Joy Division , Toto , New Order , The Bangles , Foreigner , Echo and the Bunnymen , Peter Gabriel , and Corey Hart , as well as excerpts
from Tangerine Dream , John Carpenter , and Vangelis . In particular , The Clash 's `` Should I Stay or Should I Go '' was specifically picked to play at
pivotal moments of the story , such as when Will is trying to communicate with Joyce from the Upside Down . Soundtracks Stranger Things ' original soundtrack ,
consisting of 75 songs from Stein and Dixon split across two volumes , was released by Lakeshore Records . Digital release and streaming options were released
on August 10 and 19 , 2016 for the two volumes , respectively , while retail versions were available on September 16 and 23 , 2016 .
</item 8>

<item 9>
Bad Boys (Inner Circle song)
Chart performance ( edit ) Weekly charts ( edit ) Year Chart Peak position 1991 Finland ( Suomen virallinen lista ) Norway ( VG - lista ) Sweden (
Sverigetopplistan ) Australia ( ARIA ) 25 Austria ( Ö3 Austria Top 40 ) 21 Canada Top Singles ( RPM ) 19 Germany ( Official German Charts ) 35 Ireland ( IRMA )
26 New Zealand ( Recorded Music NZ ) 5 UK Singles ( Official Charts Company ) 52 US Billboard Hot 100 8 US Billboard Hot Dance Music / Maxi - Singles Sales 15
US Billboard Hot R&B Singles 58 US Billboard Rhythmic Top 40 27 US Billboard Top 40 Mainstream 7 2006 US Billboard Hot Ringtones 10 2007 US Billboard Hot
Ringtones 20 Year - end charts ( edit ) Chart ( 1993 ) Rank New Zealand ( Recorded Music NZ ) 15 US Billboard Hot 100 69 In popular culture ( edit ) `` Bad Boys
'' was selected as the theme song to Cops because a field producer for the show happened to be a fan of Inner Circle .
</item 9>

<item 10>
NBC Sunday Night Football
Singer Pink sang the theme song for the broadcast in 2006 , a reworking of the Joan Jett song `` I Hate Myself for Loving You '' retitled `` Waiting All Day for
Sunday Night '' . Several alternative versions were used throughout the season , substituting different lyrics when appropriate ( such as `` Waiting All Year
For Opening Night '' ) . In 2007 , country singer Faith Hill replaced Pink as the singer of the opening theme , and a new arrangement of the Joan Jett song
coincided with her debut . The Faith Hill intro , in particular , was not without criticism and parody . The intro was lampooned in the October 9 , 2010 episode
of Saturday Night Live , with host Jane Lynch as Hill ( with Jason Sudeikis as Al Michaels and Bill Hader as Cris Collinsworth ) . In the 30 Rock episode ``
Season 4 '' , the character of Jenna Maroney ( Jane Krakowski ) sings what appears to be an allusion to the Faith Hill intro for NBC 's fictional Tennis Night
in America program .
</item 10>
=== END ITEM id=fb54cd722d5f925e ===

=== ITEM id=717b309ec049f169 ===
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
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
What Ifs
The song was released with the album through RCA Nashville and was serviced to radio as the third single on February 6 , 2017 . Contents ( hide ) 1 Background 2
Composition 3 Commercial performance 4 Music video 5 Chart performance 5.1 Weekly charts 5.2 Year - end charts 6 Certifications and sales 7 References
Background ( edit ) The song was written by Brown with Matthew McGinn and Jordan Schmidt . The song is about a couple where the question `` what if '' is asked
repeatedly about what would happen in their relationship . It features a duet with Lauren Alaina , a school friend from Lakeview - Fort Oglethorpe High School
when he was living in Fort Oglethorpe , Georgia . According to Brown , the song was initially not intended to be a duet , he said : `` Once we decided on adding
a woman to `` What Ifs , '' I knew that I wanted Lauren to sing it , and , of course , she killed it . ''
</item 2>

<item 3>
Graph query row:
entity: official music video for What Ifs
relationship: LOCATED_IN
related: Californian coast
</item 3>

<item 4>
What Ifs
According to Brown , another song was going to be used in the album , however , he was asked so much about `` What Ifs '' on social media that it was then
decided that the song should be included in the album . Composition ( edit ) The song is in the key of G minor with a moderately fast tempo of approximately 126
beats per minute . It follows the chord progression Gm - B ♭ - F. Brown 's vocals range from F - B ♭ . Commercial performance ( edit ) The song sold 6,000
downloads in the US the week the album was released on December 2 , 2016 , allowing it to chart at number 37 on the US Billboard Hot Country Songs chart . The
song first entered the US Billboard Country Airplay chart for the week of March 4 , 2017 at number 60 . It reached No. 1 on Hot Country Songs in October 2017 ,
ending the record breaking 34 - week reign of Sam Hunt 's `` Body Like a Back Road '' . The song is the first No. 1 on the chart for both Brown and Alaina .
</item 4>

<item 5>
What Ifs
Jump up ^ Bjorke , Matt ( January 3 , 2018 ) . `` Top 30 Country Digital Singles Chart : January 3 , 2018 '' . Roughstock . Retrieved January 14 , 2017 . ^ Jump
up to : Bjorke , Matt ( March 6 , 2018 ) . `` The Top 30 Digital Country Singles : March 5 , 2018 '' . Roughstock . Retrieved March 11 , 2018 . Jump up ^ ``
Watch Kane Brown , Lauren Alaina 's Picturesque ' What Ifs ' Video '' . Rolling Stone . May 15 , 2017 . Jump up ^ Kimble , Lindsay ( May 4 , 2017 ) . `` From
Middle School Chorus to Nashville , Kane Brown and Lauren Alaina Team Up for Duet ' What Ifs ' '' . Jump up ^ `` Kane Brown Chart History ( Canadian Hot 100 )
'' . Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard .
</item 5>

<item 6>
What Ifs
What Ifs - wikipedia What Ifs Jump to : navigation , search `` What Ifs '' Single by Kane Brown featuring Lauren Alaina from the album Kane Brown Released
February 6 , 2017 ( 2017 - 02 - 06 ) Format Digital download Genre Country Length 3 : 08 Label RCA Nashville Songwriter ( s ) Kane Brown Matthew McGinn Jordan
Schmidt Producer ( s ) Dann Huff Kane Brown singles chronology `` Thunder in the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) `` Thunder in
the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) Lauren Alaina singles chronology `` Road Less Traveled '' ( 2016 ) Road Less Traveled 2016 ``
What Ifs '' ( 2017 ) What Ifs2017 `` Doin ' Fine '' ( 2017 ) Doin ' Fine 2017 `` What Ifs '' is a song recorded by American country music singer Kane Brown
featuring Lauren Alaina for his self - titled debut album .
</item 6>

<item 7>
What Ifs
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
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
Graph query row:
entity: official music video for What Ifs
relationship: LOCATED_IN
related: The Inn at Newport Ranch
</item 9>

<item 10>
What Ifs
wikipedia.org/w/index.php?title=What_Ifs&oldid=838232791 '' Categories : 2016 songs 2017 singles Kane Brown songs Lauren Alaina songs RCA Records Nashville
singles Vocal duets Song recordings produced by Dann Huff Songs written by Jordan Schmidt Billboard Hot Country Songs number - one singles Hidden categories :
Music infoboxes with deprecated parameters Articles with hAudio microformats Singlechart usages for Canada Singlechart called without song Singlechart usages
for Billboardcanadacountry Singlechart usages for Billboardhot100 Singlechart usages for Billboardcountryairplay Singlechart usages for Billboardcountrysongs
Certification Table Entry usages for Canada Certification Table Entry usages for United States Talk Contents About Wikipedia Español Edit links This page was
last edited on 25 April 2018 , at 18 : 59 . About Wikipedia
</item 10>
=== END ITEM id=717b309ec049f169 ===

=== ITEM id=9c5ac083f4d26d5c ===
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
PMID 8064269 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Superfecundation&oldid=818981440 '' Categories : Reproduction Twin Triplets Multiple
births Free sex Sibling Hidden categories : Articles needing more detailed references All articles with dead external links Articles with dead external links
from June 2017 Articles needing additional references from May 2009 All articles needing additional references Talk Contents About Wikipedia Deutsch فارسی
Français Nederlands Polski Português Русский Slovenčina Edit links This page was last edited on 6 January 2018 , at 18 : 28 . About Wikipedia
</item 2>

<item 3>
Superfecundation
PMID 26548710 . Jump up ^ Wenk , R.E. ; Houtz , T ; Brooks , M ; Chiafari , F.A. ( 1992 ) . `` How frequent is heteropaternal superfecundation ? '' . Acta
geneticae medicae et gemellologiae. 41 ( 1 ) : 43 -- 7 . PMID 1488855 .
</item 3>

<item 4>
Superfecundation
This led to the discovery that the twins had different fathers . In 1982 , twins who were born with two different skin colors were discovered to be conceived as
a result of heteropaternal superfecundation . In 2001 , a case of spontaneous monopaternal superfecundation was reported after a woman undergoing IVF treatments
gave birth to quintuplets after only two embryos were implanted . Genetic testing supported that the twinning was not a result of the embryos splitting , and
that all five boys shared the same father . In 2016 , an IVF - implanted surrogate mother gave birth to two children : one genetically unrelated child from an
implanted embryo , and a biological child from her own egg and her husband 's sperm . See also ( edit ) Castor and Pollux , said to be twins having different
fathers Chimera ( genetics ) Mixed twins Polyspermy References ( edit ) Jump up ^ Worland , Justin ( May 8 , 2015 ) . `` The Science of How Women Can Have Twins
With 2 Different Fathers '' . Time . Retrieved March 23 , 2017 .
</item 4>

<item 5>
Superfecundation
PMID 1488855 . Jump up ^ Fierro , Pamela Prindle ( January 5 , 2016 ) , `` Heteropaternal Superfecundation - Twins with Different Fathers '' , About.com ,
archived from the original on April 4 , 2016 Jump up ^
https://www.scopus.com/record/display.uri?eid=2-s2.0-0020008620&origin=inward&txGid=F6227767DB3CDB724EA92EFC60A5A985.wsnAw8kcdt7IPYLO0V48gA%3a2 Jump up ^
Amsalem , Hagai ; Tsvieli , Rimona ; Zentner , Bat Sheva ; Yagel , Simcha ; Mitrani - Rosenbaum , Stella ; Hurwitz , Arye ( 2001 ) . `` Monopaternal
superfecundation of quintuplets after transfer of two embryos in an in vitro fertilization cycle '' . Fertility and Sterility. 76 ( 3 ) : 621 -- 3 .
</item 5>

<item 6>
Superfecundation
Fertility and Sterility. 76 ( 3 ) : 621 -- 3 . doi : 10.1016 / S0015 - 0282 ( 01 ) 01976 - 8 . PMID 11532493 . Jump up ^ Susan Scutti ( November 3 , 2017 ) . ``
Surrogate mom delivers two babies , one her own '' . CNN . Further reading ( edit ) `` Doctors Discover Dallas Twins Have Two Different Dads '' , Fox News , Sky
News , 2009 Lu , H.L. ; Wang , C.X. ; Wu , F.Q. ; Li , J.J. ( 1994 ) . `` Paternity identification in twins with different fathers '' . Journal of forensic
sciences . 39 ( 4 ) : 1100 -- 2 . PMID 8064269 .
</item 6>

<item 7>
Human fertilization
In a reference group representing all women , the 95 % prediction interval of the LMP - to - ovulation is 8.2 to 20.5 days . The average time to birth has been
estimated to be 268 days ( 38 weeks and two days ) from ovulation , with a standard deviation of 10 days or coefficient of variation of 3.7 % . Fertilization
age is sometimes used postnatally ( after birth ) as well to estimate various risk factors . For example , it is a better predictor than postnatal age for risk
of intraventricular hemorrhage in premature babies treated with extracorporeal membrane oxygenation . Diseases ( edit ) Various disorders can arise from defects
in the fertilization process . Polyspermy results from multiple sperm fertilizing an egg . However , some researchers have found that in rare pairs of fraternal
twins , their origin might have been from the fertilization of one egg cell from the mother and eight sperm cells from the father . This possibility has been
investigated by computer simulations of the fertilization process .
</item 7>

<item 8>
Superfecundation
This is known as superfetation . Heteropaternal superfecundation in mammals ( edit ) Heteropaternal superfecundation is common in animals such as cats and dogs
. Stray dogs can produce litters in which every puppy has a different sire . Though rare in humans , cases have been documented . In one study on humans , the
frequency was 2.4 % among dizygotic twins whose parents had been involved in paternity suits . Cases involving superfecundation ( edit ) Koen and Teun Stuart ,
Dutch boys who were the result of in vitro fertilization ( IVF ) . In a mixup at the laboratory , equipment had been used twice , causing another man 's sperm
to be mixed with the intended father 's . In 1995 , a young woman gave birth to diamniotic monochorionic twins , who were originally assumed to be monozygotic
twins until a paternity suit led to a DNA test . This led to the discovery that the twins had different fathers .
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
Superfecundation - wikipedia Superfecundation Jump to : navigation , search This article needs additional citations for verification . Please help improve this
article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2009 ) ( Learn how and when to remove this template
message ) Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to
twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal
superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two
separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning ,
genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through
artificial insemination .
</item 10>
=== END ITEM id=9c5ac083f4d26d5c ===

=== ITEM id=a3be46bbda174b90 ===
Gold sentences and facts:
Sentence 0: Country Adult prevalence of HIV / AIDS Number of people with HIV / AIDS Annual deaths from HIV / AIDS Year of estimate Swaziland 27.20 % 220,000 3,900 2016 Lesotho 25.00 % 330,000 9,900 2016 Botswana 21.90 % 360,000 3,900 2016 South Africa 18.90 % 7,100,000 110,000 2016 Namibia 13.80 % 230,000 4,300 2016 Zimbabwe 13.50 % 1,300,000 30,000 2016 Zambia 12.40 % 1,200,000 21,000 2016 Mozambique 12.30 % 1,800,000 62,000 2016 Malawi 9.20 % 1,000,000 24,000 2016 Uganda 6.50 % 1,400,000 28,000 2016 Equatorial Guinea 6.20 % 35,000 - 2016 Kenya 5.40 % 1,600,000 36,000 2016 Tanzania 4.70 % 1,400,000 33,000 2016 Central African Republic 4.00 % 130,000 7,300 2016 Cameroon 3.80 % 560,000 29,000 2016 Gabon 3.60 % 48,000 1,500 2016 Bahamas , The 3.30 % 8,200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000 160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016 Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000 1,300 2016 Liberia 1.60 % 43,000 2,800 2016 Guyana 1.60 % 8,500 - 2016 Ghana 1.60 % 290,000 15,000 2016 Guinea 1.50 % 120,000 5,800 2016 Suriname 1.40 % 4,900 - 2016 Barbados 1.30 % 2,600 - 2016 Djibouti 1.30 % 8,600 - 2016 Chad 1.30 % 110,000 2,800 2016 Trinidad and Tobago 1.20 % 11,000 - 2016 Thailand 1.10 % 450,000 16,000 2016 Ethiopia 1.10 % 710,000 20,000 2016 Burundi 1.10 % 84,000 2,900 2016 Benin 1.00 % 67,000 2,400 2016 Mali 1.00 % 110,000 6,100 2016 Dominican Republic 1.00 % 67,000 2,200 2016 Ukraine 0.90 % 240,000 8,500 2016 Papua New Guinea 0.90 % 46,000 1,100 2016 Cabo Verde 0.80 % 2,800 - 2016 Panama 0.80 % 21,000 - 2016 Burma 0.80 % 230,000 7,800 2016 Burkina Faso 0.80 % 95,000 3,100 2016 Latvia 0.70 % 6,600 - 2016 Congo , Democratic Republic of 0.70 % 370,000 19,000 2016 Cambodia 0.60 % 71,000 1,800 2016 Venezuela 0.60 % 120,000 2,500 2016 Moldova 0.60 % 15,000 - 2016 Eritrea 0.60 % 15,000 - 2016 El Salvador 0.60 % 24,000 - 2016 Uruguay 0.60 % 12,000 - 2016 Brazil 0.60 % 830,000 14,000 2016 Georgia 0.50 % 12,000 - 2016 Mauritania 0.50 % 11,000 - 2016 Chile 0.50 % 61,000 - 2016 Guatemala 0.50 % 46,000 1,600 2016 Paraguay 0.50 % 19,000 - 2016 Niger 0.40 % 48,000 3,400 2016 Malaysia 0.40 % 97,000 7,000 2016 Indonesia 0.40 % 620,000 38,000 2016 Honduras 0.40 % 21,000 - 2016 Canada 0.40 % 180,000 - 2016 Cuba 0.40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016 Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000 - 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico 0.30 % 220,000 4,200 2016 Armenia 0.20 % 3,300 - 2016 Ireland 0.20 % 6,200 - 2016 Kazakhstan 0.20 % 26,000 - 2016 Kyrgyzstan 0.20 % 8,500 - 2016 Lithuania 0.20 % 2,900 - 2016 Madagascar 0.20 % 31,000 1,600 2016 Nepal 0.20 % 32,000 1,700 2016 Netherlands 0.20 % 23,000 - 2016 Nicaragua 0.20 % 8,900 - 2016 Sudan 0.20 % 56,000 3,000 2016 Sweden 0.20 % 11,000 - 2016 Albania 0.10 % 1,700 - 2016 Australia 0.10 % 25,000 - 2016 Fiji 0.10 % - - 2016 Taiwan 0.10 % 31,620 - 2016 Azerbaijan 0.10 % 9,200 - 2016 Iran 0.10 % 66,000 4,000 2016 Romania 0.10 % 16,000 - 2016 Montenegro 0.10 % - - 2016 Morocco 0.10 % 22,000 - 2016 Pakistan 0.10 % 130,000 5,500 2016 Philippines 0.10 % 56,000 - 2016 Malta 0.10 % - - 2016 Algeria - 13,000 - 2016 Bangladesh - 12,000 1,000 2016 Egypt - 11,000 - 2016 Yemen - 9,900 - 2016 Saudi Arabia - 8,200 - 2016 Afghanistan - 7,500 - 2016 Sri Lanka - 4,000 - 2016 Bulgaria - 3,500 - 2016 Czech Republic - 3,400 - 2016 Tunisia - 2,900 - 2016 Serbia - 2,700 - 2016 Lebanon - 2,200 - 2016 Croatia - 1,500 - 2016 Bosnia 0.009 % 350 - 2017
  Fact 0.0: The table gives, for each country, the adult prevalence of HIV/AIDS, the number of people with HIV/AIDS, the annual deaths from HIV/AIDS, and the year of estimate.
  Fact 0.1: The year of estimate for Somalia is 2016.
  Fact 0.2: In 2016, the adult HIV/AIDS prevalence in Swaziland was 27.20%.
  Fact 0.3: In 2016, Swaziland had 220,000 people with HIV/AIDS.
  Fact 0.4: In 2016, there were 3,900 annual deaths from HIV/AIDS in Swaziland.
  Fact 0.5: In 2016, the adult HIV/AIDS prevalence in Lesotho was 25.00%.
  Fact 0.6: In 2016, Lesotho had 330,000 people with HIV/AIDS.
  Fact 0.7: In 2016, there were 9,900 annual deaths from HIV/AIDS in Lesotho.
  Fact 0.8: In 2016, the adult HIV/AIDS prevalence in Botswana was 21.90%.
  Fact 0.9: In 2016, Botswana had 360,000 people with HIV/AIDS.
  Fact 0.10: In 2016, there were 3,900 annual deaths from HIV/AIDS in Botswana.
  Fact 0.11: In 2016, the adult HIV/AIDS prevalence in South Africa was 18.90%.
  Fact 0.12: In 2016, South Africa had 7,100,000 people with HIV/AIDS.
  Fact 0.13: In 2016, there were 110,000 annual deaths from HIV/AIDS in South Africa.
  Fact 0.14: In 2016, the adult HIV/AIDS prevalence in Namibia was 13.80%.
  Fact 0.15: In 2016, Namibia had 230,000 people with HIV/AIDS.
  Fact 0.16: In 2016, there were 4,300 annual deaths from HIV/AIDS in Namibia.
  Fact 0.17: In 2016, the adult HIV/AIDS prevalence in Zimbabwe was 13.50%.
  Fact 0.18: In 2016, Zimbabwe had 1,300,000 people with HIV/AIDS.
  Fact 0.19: In 2016, there were 30,000 annual deaths from HIV/AIDS in Zimbabwe.
  Fact 0.20: In 2016, the adult HIV/AIDS prevalence in Zambia was 12.40%.
  Fact 0.21: In 2016, Zambia had 1,200,000 people with HIV/AIDS.
  Fact 0.22: In 2016, there were 21,000 annual deaths from HIV/AIDS in Zambia.
  Fact 0.23: In 2016, the adult HIV/AIDS prevalence in Mozambique was 12.30%.
  Fact 0.24: In 2016, Mozambique had 1,800,000 people with HIV/AIDS.
  Fact 0.25: In 2016, there were 62,000 annual deaths from HIV/AIDS in Mozambique.
  Fact 0.26: In 2016, the adult HIV/AIDS prevalence in Malawi was 9.20%.
  Fact 0.27: In 2016, Malawi had 1,000,000 people with HIV/AIDS.
  Fact 0.28: In 2016, there were 24,000 annual deaths from HIV/AIDS in Malawi.
  Fact 0.29: In 2016, the adult HIV/AIDS prevalence in Uganda was 6.50%.
  Fact 0.30: In 2016, Uganda had 1,400,000 people with HIV/AIDS.
  Fact 0.31: In 2016, there were 28,000 annual deaths from HIV/AIDS in Uganda.
  Fact 0.32: In 2016, the adult HIV/AIDS prevalence in Equatorial Guinea was 6.20%.
  Fact 0.33: In 2016, Equatorial Guinea had 35,000 people with HIV/AIDS.
  Fact 0.34: In 2016, the adult HIV/AIDS prevalence in Kenya was 5.40%.
  Fact 0.35: In 2016, Kenya had 1,600,000 people with HIV/AIDS.
  Fact 0.36: In 2016, there were 36,000 annual deaths from HIV/AIDS in Kenya.
  Fact 0.37: In 2016, the adult HIV/AIDS prevalence in Tanzania was 4.70%.
  Fact 0.38: In 2016, Tanzania had 1,400,000 people with HIV/AIDS.
  Fact 0.39: In 2016, there were 33,000 annual deaths from HIV/AIDS in Tanzania.
  Fact 0.40: In 2016, the adult HIV/AIDS prevalence in Central African Republic was 4.00%.
  Fact 0.41: In 2016, Central African Republic had 130,000 people with HIV/AIDS.
  Fact 0.42: In 2016, there were 7,300 annual deaths from HIV/AIDS in Central African Republic.
  Fact 0.43: In 2016, the adult HIV/AIDS prevalence in Cameroon was 3.80%.
  Fact 0.44: In 2016, Cameroon had 560,000 people with HIV/AIDS.
  Fact 0.45: In 2016, there were 29,000 annual deaths from HIV/AIDS in Cameroon.
  Fact 0.46: In 2016, the adult HIV/AIDS prevalence in Gabon was 3.60%.
  Fact 0.47: In 2016, Gabon had 48,000 people with HIV/AIDS.
  Fact 0.48: In 2016, there were 1,500 annual deaths from HIV/AIDS in Gabon.
  Fact 0.49: In 2016, the adult HIV/AIDS prevalence in Bahamas, The was 3.30%.
  Fact 0.50: In 2016, Bahamas, The had 8,200 people with HIV/AIDS.
  Fact 0.51: In 2016, the adult HIV/AIDS prevalence in Rwanda was 3.10%.
  Fact 0.52: In 2016, Rwanda had 220,000 people with HIV/AIDS.
  Fact 0.53: In 2016, there were 3,300 annual deaths from HIV/AIDS in Rwanda.
  Fact 0.54: In 2016, the adult HIV/AIDS prevalence in Congo, Republic of the was 3.10%.
  Fact 0.55: In 2016, Congo, Republic of the had 91,000 people with HIV/AIDS.
  Fact 0.56: In 2016, there were 3,800 annual deaths from HIV/AIDS in Congo, Republic of the.
  Fact 0.57: In 2016, the adult HIV/AIDS prevalence in Guinea-Bissau was 3.10%.
  Fact 0.58: In 2016, Guinea-Bissau had 36,000 people with HIV/AIDS.
  Fact 0.59: In 2016, there were 2,000 annual deaths from HIV/AIDS in Guinea-Bissau.
  Fact 0.60: In 2016, the adult HIV/AIDS prevalence in Nigeria was 2.90%.
  Fact 0.61: In 2016, Nigeria had 3,200,000 people with HIV/AIDS.
  Fact 0.62: In 2016, there were 160,000 annual deaths from HIV/AIDS in Nigeria.
  Fact 0.63: In 2016, the adult HIV/AIDS prevalence in Cote d'Ivoire was 2.70%.
  Fact 0.64: In 2016, Cote d'Ivoire had 460,000 people with HIV/AIDS.
  Fact 0.65: In 2016, there were 25,000 annual deaths from HIV/AIDS in Cote d'Ivoire.
  Fact 0.66: In 2016, the adult HIV/AIDS prevalence in South Sudan was 2.70%.
  Fact 0.67: In 2016, South Sudan had 200,000 people with HIV/AIDS.
  Fact 0.68: In 2016, there were 13,000 annual deaths from HIV/AIDS in South Sudan.
  Fact 0.69: In 2016, the adult HIV/AIDS prevalence in Haiti was 2.10%.
  Fact 0.70: In 2016, Haiti had 150,000 people with HIV/AIDS.
  Fact 0.71: In 2016, there were 4,600 annual deaths from HIV/AIDS in Haiti.
  Fact 0.72: In 2016, the adult HIV/AIDS prevalence in Togo was 2.10%.
  Fact 0.73: In 2016, Togo had 100,000 people with HIV/AIDS.
  Fact 0.74: In 2016, there were 5,100 annual deaths from HIV/AIDS in Togo.
  Fact 0.75: In 2016, the adult HIV/AIDS prevalence in Angola was 1.90%.
  Fact 0.76: In 2016, Angola had 280,000 people with HIV/AIDS.
  Fact 0.77: In 2016, there were 11,000 annual deaths from HIV/AIDS in Angola.
  Fact 0.78: In 2016, the adult HIV/AIDS prevalence in Belize was 1.80%.
  Fact 0.79: In 2016, Belize had 4,300 people with HIV/AIDS.
  Fact 0.80: In 2016, the adult HIV/AIDS prevalence in Sierra Leone was 1.70%.
  Fact 0.81: In 2016, Sierra Leone had 67,000 people with HIV/AIDS.
  Fact 0.82: In 2016, there were 2,800 annual deaths from HIV/AIDS in Sierra Leone.
  Fact 0.83: In 2016, the adult HIV/AIDS prevalence in Gambia, The was 1.70%.
  Fact 0.84: In 2016, Gambia, The had 20,000 people with HIV/AIDS.
  Fact 0.85: In 2016, there were 1,100 annual deaths from HIV/AIDS in Gambia, The.
  Fact 0.86: In 2016, the adult HIV/AIDS prevalence in Jamaica was 1.70%.
  Fact 0.87: In 2016, Jamaica had 30,000 people with HIV/AIDS.
  Fact 0.88: In 2016, there were 1,300 annual deaths from HIV/AIDS in Jamaica.
  Fact 0.89: In 2016, the adult HIV/AIDS prevalence in Liberia was 1.60%.
  Fact 0.90: In 2016, Liberia had 43,000 people with HIV/AIDS.
  Fact 0.91: In 2016, there were 2,800 annual deaths from HIV/AIDS in Liberia.
  Fact 0.92: In 2016, the adult HIV/AIDS prevalence in Guyana was 1.60%.
  Fact 0.93: In 2016, Guyana had 8,500 people with HIV/AIDS.
  Fact 0.94: In 2016, the adult HIV/AIDS prevalence in Ghana was 1.60%.
  Fact 0.95: In 2016, Ghana had 290,000 people with HIV/AIDS.
  Fact 0.96: In 2016, there were 15,000 annual deaths from HIV/AIDS in Ghana.
  Fact 0.97: In 2016, the adult HIV/AIDS prevalence in Guinea was 1.50%.
  Fact 0.98: In 2016, Guinea had 120,000 people with HIV/AIDS.
  Fact 0.99: In 2016, there were 5,800 annual deaths from HIV/AIDS in Guinea.
  Fact 0.100: In 2016, the adult HIV/AIDS prevalence in Suriname was 1.40%.
  Fact 0.101: In 2016, Suriname had 4,900 people with HIV/AIDS.
  Fact 0.102: In 2016, the adult HIV/AIDS prevalence in Barbados was 1.30%.
  Fact 0.103: In 2016, Barbados had 2,600 people with HIV/AIDS.
  Fact 0.104: In 2016, the adult HIV/AIDS prevalence in Djibouti was 1.30%.
  Fact 0.105: In 2016, Djibouti had 8,600 people with HIV/AIDS.
  Fact 0.106: In 2016, the adult HIV/AIDS prevalence in Chad was 1.30%.
  Fact 0.107: In 2016, Chad had 110,000 people with HIV/AIDS.
  Fact 0.108: In 2016, there were 2,800 annual deaths from HIV/AIDS in Chad.
  Fact 0.109: In 2016, the adult HIV/AIDS prevalence in Trinidad and Tobago was 1.20%.
  Fact 0.110: In 2016, Trinidad and Tobago had 11,000 people with HIV/AIDS.
  Fact 0.111: In 2016, the adult HIV/AIDS prevalence in Thailand was 1.10%.
  Fact 0.112: In 2016, Thailand had 450,000 people with HIV/AIDS.
  Fact 0.113: In 2016, there were 16,000 annual deaths from HIV/AIDS in Thailand.
  Fact 0.114: In 2016, the adult HIV/AIDS prevalence in Ethiopia was 1.10%.
  Fact 0.115: In 2016, Ethiopia had 710,000 people with HIV/AIDS.
  Fact 0.116: In 2016, there were 20,000 annual deaths from HIV/AIDS in Ethiopia.
  Fact 0.117: In 2016, the adult HIV/AIDS prevalence in Burundi was 1.10%.
  Fact 0.118: In 2016, Burundi had 84,000 people with HIV/AIDS.
  Fact 0.119: In 2016, there were 2,900 annual deaths from HIV/AIDS in Burundi.
  Fact 0.120: In 2016, the adult HIV/AIDS prevalence in Benin was 1.00%.
  Fact 0.121: In 2016, Benin had 67,000 people with HIV/AIDS.
  Fact 0.122: In 2016, there were 2,400 annual deaths from HIV/AIDS in Benin.
  Fact 0.123: In 2016, the adult HIV/AIDS prevalence in Mali was 1.00%.
  Fact 0.124: In 2016, Mali had 110,000 people with HIV/AIDS.
  Fact 0.125: In 2016, there were 6,100 annual deaths from HIV/AIDS in Mali.
  Fact 0.126: In 2016, the adult HIV/AIDS prevalence in Dominican Republic was 1.00%.
  Fact 0.127: In 2016, Dominican Republic had 67,000 people with HIV/AIDS.
  Fact 0.128: In 2016, there were 2,200 annual deaths from HIV/AIDS in Dominican Republic.
  Fact 0.129: In 2016, the adult HIV/AIDS prevalence in Ukraine was 0.90%.
  Fact 0.130: In 2016, Ukraine had 240,000 people with HIV/AIDS.
  Fact 0.131: In 2016, there were 8,500 annual deaths from HIV/AIDS in Ukraine.
  Fact 0.132: In 2016, the adult HIV/AIDS prevalence in Papua New Guinea was 0.90%.
  Fact 0.133: In 2016, Papua New Guinea had 46,000 people with HIV/AIDS.
  Fact 0.134: In 2016, there were 1,100 annual deaths from HIV/AIDS in Papua New Guinea.
  Fact 0.135: In 2016, the adult HIV/AIDS prevalence in Cabo Verde was 0.80%.
  Fact 0.136: In 2016, Cabo Verde had 2,800 people with HIV/AIDS.
  Fact 0.137: In 2016, the adult HIV/AIDS prevalence in Panama was 0.80%.
  Fact 0.138: In 2016, Panama had 21,000 people with HIV/AIDS.
  Fact 0.139: In 2016, the adult HIV/AIDS prevalence in Burma was 0.80%.
  Fact 0.140: In 2016, Burma had 230,000 people with HIV/AIDS.
  Fact 0.141: In 2016, there were 7,800 annual deaths from HIV/AIDS in Burma.
  Fact 0.142: In 2016, the adult HIV/AIDS prevalence in Burkina Faso was 0.80%.
  Fact 0.143: In 2016, Burkina Faso had 95,000 people with HIV/AIDS.
  Fact 0.144: In 2016, there were 3,100 annual deaths from HIV/AIDS in Burkina Faso.
  Fact 0.145: In 2016, the adult HIV/AIDS prevalence in Latvia was 0.70%.
  Fact 0.146: In 2016, Latvia had 6,600 people with HIV/AIDS.
  Fact 0.147: In 2016, the adult HIV/AIDS prevalence in Congo, Democratic Republic of was 0.70%.
  Fact 0.148: In 2016, Congo, Democratic Republic of had 370,000 people with HIV/AIDS.
  Fact 0.149: In 2016, there were 19,000 annual deaths from HIV/AIDS in Congo, Democratic Republic of.
  Fact 0.150: In 2016, the adult HIV/AIDS prevalence in Cambodia was 0.60%.
  Fact 0.151: In 2016, Cambodia had 71,000 people with HIV/AIDS.
  Fact 0.152: In 2016, there were 1,800 annual deaths from HIV/AIDS in Cambodia.
  Fact 0.153: In 2016, the adult HIV/AIDS prevalence in Venezuela was 0.60%.
  Fact 0.154: In 2016, Venezuela had 120,000 people with HIV/AIDS.
  Fact 0.155: In 2016, there were 2,500 annual deaths from HIV/AIDS in Venezuela.
  Fact 0.156: In 2016, the adult HIV/AIDS prevalence in Moldova was 0.60%.
  Fact 0.157: In 2016, Moldova had 15,000 people with HIV/AIDS.
  Fact 0.158: In 2016, the adult HIV/AIDS prevalence in Eritrea was 0.60%.
  Fact 0.159: In 2016, Eritrea had 15,000 people with HIV/AIDS.
  Fact 0.160: In 2016, the adult HIV/AIDS prevalence in El Salvador was 0.60%.
  Fact 0.161: In 2016, El Salvador had 24,000 people with HIV/AIDS.
  Fact 0.162: In 2016, the adult HIV/AIDS prevalence in Uruguay was 0.60%.
  Fact 0.163: In 2016, Uruguay had 12,000 people with HIV/AIDS.
  Fact 0.164: In 2016, the adult HIV/AIDS prevalence in Brazil was 0.60%.
  Fact 0.165: In 2016, Brazil had 830,000 people with HIV/AIDS.
  Fact 0.166: In 2016, there were 14,000 annual deaths from HIV/AIDS in Brazil.
  Fact 0.167: In 2016, the adult HIV/AIDS prevalence in Georgia was 0.50%.
  Fact 0.168: In 2016, Georgia had 12,000 people with HIV/AIDS.
  Fact 0.169: In 2016, the adult HIV/AIDS prevalence in Mauritania was 0.50%.
  Fact 0.170: In 2016, Mauritania had 11,000 people with HIV/AIDS.
  Fact 0.171: In 2016, the adult HIV/AIDS prevalence in Chile was 0.50%.
  Fact 0.172: In 2016, Chile had 61,000 people with HIV/AIDS.
  Fact 0.173: In 2016, the adult HIV/AIDS prevalence in Guatemala was 0.50%.
  Fact 0.174: In 2016, Guatemala had 46,000 people with HIV/AIDS.
  Fact 0.175: In 2016, there were 1,600 annual deaths from HIV/AIDS in Guatemala.
  Fact 0.176: In 2016, the adult HIV/AIDS prevalence in Paraguay was 0.50%.
  Fact 0.177: In 2016, Paraguay had 19,000 people with HIV/AIDS.
  Fact 0.178: In 2016, the adult HIV/AIDS prevalence in Niger was 0.40%.
  Fact 0.179: In 2016, Niger had 48,000 people with HIV/AIDS.
  Fact 0.180: In 2016, there were 3,400 annual deaths from HIV/AIDS in Niger.
  Fact 0.181: In 2016, the adult HIV/AIDS prevalence in Malaysia was 0.40%.
  Fact 0.182: In 2016, Malaysia had 97,000 people with HIV/AIDS.
  Fact 0.183: In 2016, there were 7,000 annual deaths from HIV/AIDS in Malaysia.
  Fact 0.184: In 2016, the adult HIV/AIDS prevalence in Indonesia was 0.40%.
  Fact 0.185: In 2016, Indonesia had 620,000 people with HIV/AIDS.
  Fact 0.186: In 2016, there were 38,000 annual deaths from HIV/AIDS in Indonesia.
  Fact 0.187: In 2016, the adult HIV/AIDS prevalence in Honduras was 0.40%.
  Fact 0.188: In 2016, Honduras had 21,000 people with HIV/AIDS.
  Fact 0.189: In 2016, the adult HIV/AIDS prevalence in Canada was 0.40%.
  Fact 0.190: In 2016, Canada had 180,000 people with HIV/AIDS.
  Fact 0.191: In 2016, the adult HIV/AIDS prevalence in Cuba was 0.40%.
  Fact 0.192: In 2016, Cuba had 25,000 people with HIV/AIDS.
  Fact 0.193: In 2016, the adult HIV/AIDS prevalence in Costa Rica was 0.40%.
  Fact 0.194: In 2016, Costa Rica had 13,000 people with HIV/AIDS.
  Fact 0.195: In 2016, the adult HIV/AIDS prevalence in Colombia was 0.40%.
  Fact 0.196: In 2016, Colombia had 120,000 people with HIV/AIDS.
  Fact 0.197: In 2016, there were 2,800 annual deaths from HIV/AIDS in Colombia.
  Fact 0.198: In 2016, the adult HIV/AIDS prevalence in Vietnam was 0.40%.
  Fact 0.199: In 2016, Vietnam had 250,000 people with HIV/AIDS.
  Fact 0.200: In 2016, there were 8,000 annual deaths from HIV/AIDS in Vietnam.
  Fact 0.201: In 2016, the adult HIV/AIDS prevalence in Argentina was 0.40%.
  Fact 0.202: In 2016, Argentina had 120,000 people with HIV/AIDS.
  Fact 0.203: In 2016, there were 2,400 annual deaths from HIV/AIDS in Argentina.
  Fact 0.204: In 2016, the adult HIV/AIDS prevalence in Belarus was 0.40%.
  Fact 0.205: In 2016, Belarus had 19,000 people with HIV/AIDS.
  Fact 0.206: In 2015, the adult HIV/AIDS prevalence in USA was 0.30%.
  Fact 0.207: In 2015, USA had 1,100,000 people with HIV/AIDS.
  Fact 0.208: In 2016, the adult HIV/AIDS prevalence in Senegal was 0.40%.
  Fact 0.209: In 2016, Senegal had 41,000 people with HIV/AIDS.
  Fact 0.210: In 2016, there were 1,900 annual deaths from HIV/AIDS in Senegal.
  Fact 0.211: In 2016, the adult HIV/AIDS prevalence in India was 0.30%.
  Fact 0.212: In 2016, India had 2,100,000 people with HIV/AIDS.
  Fact 0.213: In 2016, there were 62,000 annual deaths from HIV/AIDS in India.
  Fact 0.214: In 2016, the adult HIV/AIDS prevalence in Ecuador was 0.30%.
  Fact 0.215: In 2016, Ecuador had 33,000 people with HIV/AIDS.
  Fact 0.216: In 2016, the adult HIV/AIDS prevalence in Bolivia was 0.30%.
  Fact 0.217: In 2016, Bolivia had 19,000 people with HIV/AIDS.
  Fact 0.218: In 2016, the adult HIV/AIDS prevalence in Tajikistan was 0.30%.
  Fact 0.219: In 2016, Tajikistan had 14,000 people with HIV/AIDS.
  Fact 0.220: In 2016, the adult HIV/AIDS prevalence in Peru was 0.30%.
  Fact 0.221: In 2016, Peru had 70,000 people with HIV/AIDS.
  Fact 0.222: In 2016, there were 2,200 annual deaths from HIV/AIDS in Peru.
  Fact 0.223: In 2016, the adult HIV/AIDS prevalence in Italy was 0.30%.
  Fact 0.224: In 2016, Italy had 130,000 people with HIV/AIDS.
  Fact 0.225: In 2016, the adult HIV/AIDS prevalence in Laos was 0.30%.
  Fact 0.226: In 2016, Laos had 11,000 people with HIV/AIDS.
  Fact 0.227: In 2016, the adult HIV/AIDS prevalence in Mexico was 0.30%.
  Fact 0.228: In 2016, Mexico had 220,000 people with HIV/AIDS.
  Fact 0.229: In 2016, there were 4,200 annual deaths from HIV/AIDS in Mexico.
  Fact 0.230: In 2016, the adult HIV/AIDS prevalence in Armenia was 0.20%.
  Fact 0.231: In 2016, Armenia had 3,300 people with HIV/AIDS.
  Fact 0.232: In 2016, the adult HIV/AIDS prevalence in Ireland was 0.20%.
  Fact 0.233: In 2016, Ireland had 6,200 people with HIV/AIDS.
  Fact 0.234: In 2016, the adult HIV/AIDS prevalence in Kazakhstan was 0.20%.
  Fact 0.235: In 2016, Kazakhstan had 26,000 people with HIV/AIDS.
  Fact 0.236: In 2016, the adult HIV/AIDS prevalence in Kyrgyzstan was 0.20%.
  Fact 0.237: In 2016, Kyrgyzstan had 8,500 people with HIV/AIDS.
  Fact 0.238: In 2016, the adult HIV/AIDS prevalence in Lithuania was 0.20%.
  Fact 0.239: In 2016, Lithuania had 2,900 people with HIV/AIDS.
  Fact 0.240: In 2016, the adult HIV/AIDS prevalence in Madagascar was 0.20%.
  Fact 0.241: In 2016, Madagascar had 31,000 people with HIV/AIDS.
  Fact 0.242: In 2016, there were 1,600 annual deaths from HIV/AIDS in Madagascar.
  Fact 0.243: In 2016, the adult HIV/AIDS prevalence in Nepal was 0.20%.
  Fact 0.244: In 2016, Nepal had 32,000 people with HIV/AIDS.
  Fact 0.245: In 2016, there were 1,700 annual deaths from HIV/AIDS in Nepal.
  Fact 0.246: In 2016, the adult HIV/AIDS prevalence in Netherlands was 0.20%.
  Fact 0.247: In 2016, Netherlands had 23,000 people with HIV/AIDS.
  Fact 0.248: In 2016, the adult HIV/AIDS prevalence in Nicaragua was 0.20%.
  Fact 0.249: In 2016, Nicaragua had 8,900 people with HIV/AIDS.
  Fact 0.250: In 2016, the adult HIV/AIDS prevalence in Sudan was 0.20%.
  Fact 0.251: In 2016, Sudan had 56,000 people with HIV/AIDS.
  Fact 0.252: In 2016, there were 3,000 annual deaths from HIV/AIDS in Sudan.
  Fact 0.253: In 2016, the adult HIV/AIDS prevalence in Sweden was 0.20%.
  Fact 0.254: In 2016, Sweden had 11,000 people with HIV/AIDS.
  Fact 0.255: In 2016, the adult HIV/AIDS prevalence in Albania was 0.10%.
  Fact 0.256: In 2016, Albania had 1,700 people with HIV/AIDS.
  Fact 0.257: In 2016, the adult HIV/AIDS prevalence in Australia was 0.10%.
  Fact 0.258: In 2016, Australia had 25,000 people with HIV/AIDS.
  Fact 0.259: In 2016, the adult HIV/AIDS prevalence in Fiji was 0.10%.
  Fact 0.260: In 2016, the adult HIV/AIDS prevalence in Taiwan was 0.10%.
  Fact 0.261: In 2016, Taiwan had 31,620 people with HIV/AIDS.
  Fact 0.262: In 2016, the adult HIV/AIDS prevalence in Azerbaijan was 0.10%.
  Fact 0.263: In 2016, Azerbaijan had 9,200 people with HIV/AIDS.
  Fact 0.264: In 2016, the adult HIV/AIDS prevalence in Iran was 0.10%.
  Fact 0.265: In 2016, Iran had 66,000 people with HIV/AIDS.
  Fact 0.266: In 2016, there were 4,000 annual deaths from HIV/AIDS in Iran.
  Fact 0.267: In 2016, the adult HIV/AIDS prevalence in Romania was 0.10%.
  Fact 0.268: In 2016, Romania had 16,000 people with HIV/AIDS.
  Fact 0.269: In 2016, the adult HIV/AIDS prevalence in Montenegro was 0.10%.
  Fact 0.270: In 2016, the adult HIV/AIDS prevalence in Morocco was 0.10%.
  Fact 0.271: In 2016, Morocco had 22,000 people with HIV/AIDS.
  Fact 0.272: In 2016, the adult HIV/AIDS prevalence in Pakistan was 0.10%.
  Fact 0.273: In 2016, Pakistan had 130,000 people with HIV/AIDS.
  Fact 0.274: In 2016, there were 5,500 annual deaths from HIV/AIDS in Pakistan.
  Fact 0.275: In 2016, the adult HIV/AIDS prevalence in Philippines was 0.10%.
  Fact 0.276: In 2016, Philippines had 56,000 people with HIV/AIDS.
  Fact 0.277: In 2016, the adult HIV/AIDS prevalence in Malta was 0.10%.
  Fact 0.278: In 2016, Algeria had 13,000 people with HIV/AIDS.
  Fact 0.279: In 2016, Bangladesh had 12,000 people with HIV/AIDS.
  Fact 0.280: In 2016, there were 1,000 annual deaths from HIV/AIDS in Bangladesh.
  Fact 0.281: In 2016, Egypt had 11,000 people with HIV/AIDS.
  Fact 0.282: In 2016, Yemen had 9,900 people with HIV/AIDS.
  Fact 0.283: In 2016, Saudi Arabia had 8,200 people with HIV/AIDS.
  Fact 0.284: In 2016, Afghanistan had 7,500 people with HIV/AIDS.
  Fact 0.285: In 2016, Sri Lanka had 4,000 people with HIV/AIDS.
  Fact 0.286: In 2016, Bulgaria had 3,500 people with HIV/AIDS.
  Fact 0.287: In 2016, Czech Republic had 3,400 people with HIV/AIDS.
  Fact 0.288: In 2016, Tunisia had 2,900 people with HIV/AIDS.
  Fact 0.289: In 2016, Serbia had 2,700 people with HIV/AIDS.
  Fact 0.290: In 2016, Lebanon had 2,200 people with HIV/AIDS.
  Fact 0.291: In 2016, Croatia had 1,500 people with HIV/AIDS.
  Fact 0.292: In 2017, the adult HIV/AIDS prevalence in Bosnia was 0.009%.
  Fact 0.293: In 2017, Bosnia had 350 people with HIV/AIDS.

Retrieved items:
<item 1>
List of countries by HIV/AIDS adult prevalence rate
40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016
Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000
- 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico
0.30 % 220,
</item 1>

<item 2>
List of countries by HIV/AIDS adult prevalence rate
00 % 67,000 2,400 2016 Mali 1.00 % 110,000 6,100 2016 Dominican Republic 1.00 % 67,000 2,200 2016 Ukraine 0.90 % 240,000 8,500 2016 Papua New Guinea 0.90 %
46,000 1,100 2016 Cabo Verde 0.80 % 2,800 - 2016 Panama 0.80 % 21,000 - 2016 Burma 0.80 % 230,000 7,800 2016 Burkina Faso 0.80 % 95,000 3,100 2016 Latvia 0.70 %
6,600 - 2016 Congo , Democratic Republic of 0.70 % 370,000 19,000 2016 Cambodia 0.60 % 71,000 1,800 2016 Venezuela 0.60 % 120,000 2,500 2016 Moldova 0.60 %
15,000 - 2016 Eritrea 0.
</item 2>

<item 3>
List of countries by HIV/AIDS adult prevalence rate
AVERT . Retrieved 3 August 2017 . Jump up ^ http://factbook.lincon.com/geos/print/country/countrypdf_bf.pdf Jump up ^ `` UNAIDS.org - HIV and AIDS estimates (
2013 ) '' . Jump up ^ `` The Hindu - India has 3rd - highest number of HIV - infected people : UN '' . Jump up ^ `` The Ministry of Health has Detected ... '' .
Bhutan Observer online . 2011 - 08 - 01 . Archived from the original on 2011 - 10 - 28 . Retrieved 2011 - 11 - 21 . Jump up ^ `` CIA world factbook - country
comparison '' . Jump up ^ https://www.cdc.gov/hiv/statistics/overview/index.html Jump up ^ https://www.cdc.gov.tw/downloadfile.aspx?fid=40CE068DC7A0F111
External links ( edit ) HIV / AIDS Survey Indicators Database World Health Organisation ; Prevalence of HIV among adults aged 15 to 49 ( % ) by country World
Health Organisation ;
</item 3>

<item 4>
List of countries by HIV/AIDS adult prevalence rate
200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000
160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016
Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000
1,
</item 4>

<item 5>
List of countries by HIV/AIDS adult prevalence rate
30 % 11,000 - 2016 Mexico 0.30 % 220,000 4,200 2016 Armenia 0.20 % 3,300 - 2016 Ireland 0.20 % 6,200 - 2016 Kazakhstan 0.20 % 26,000 - 2016 Kyrgyzstan 0.20 %
8,500 - 2016 Lithuania 0.20 % 2,900 - 2016 Madagascar 0.20 % 31,000 1,600 2016 Nepal 0.20 % 32,000 1,700 2016 Netherlands 0.20 % 23,000 - 2016 Nicaragua 0.20 %
8,900 - 2016 Sudan 0.20 % 56,000 3,000 2016 Sweden 0.20 % 11,000 - 2016 Albania 0.10 % 1,700 - 2016 Australia 0.10 % 25,000 - 2016 Fiji 0.10 % - - 2016 Taiwan
0.10 % 31,
</item 5>

<item 6>
List of countries by HIV/AIDS adult prevalence rate
Number of deaths due to AIDS by country World Health Organisation ; Number of people ( all ages ) living with HIV by country ( show ) HIV / AIDS topics ( show )
HIV / AIDS HIV HIV structure and genome subtypes CDC classification disease progression rates HIV / AIDS diagnosis management pathophysiology prevention
research vaccination PrEP WHO disease staging system for HIV infection and disease Children Teens / Adults Countries by AIDS prevalence rate Conditions Signs
and symptoms AIDS - defining clinical condition Diffuse infiltrative lymphocytosis syndrome Lipodystrophy Nephropathy Neurocognitive disorders Pruritus
Superinfection Tuberculosis co-infection HIV Drug Resistance Database Innate resistance to HIV History History Epidemiology Multiple sex partners Timeline AIDS
Museum Timothy Ray Brown Social AIDS orphan Catholic Church and HIV / AIDS Circumcision and HIV Criminal transmission Discrimination against people Economic
impact HIV - affected community HIV / AIDS denialism Safe sex Sex education List of HIV - positive people People With AIDS Self - Empowerment Movement Culture
International AIDS Conference International AIDS Society Joint United Nations Programme on HIV / AIDS ( UNAIDS ) President 's Emergency Plan for AIDS Relief (
PEPFAR ) Treatment Action Campaign World AIDS Day Media portrayal of HIV / AIDS Misconceptions about HIV /
</item 6>

<item 7>
Sub-Saharan Africa
In response , a number of initiatives have been launched to educate the public on HIV / AIDS . Among these are combination prevention programmes , considered to
be the most effective initiative , the abstinence , be faithful , use a condom campaign , and the Desmond Tutu HIV Foundation 's outreach programs . According
to a 2013 special report issued by the Joint United Nations Programme on HIV / AIDS ( UNAIDS ) , the number of HIV positive people in Africa receiving
anti-retroviral treatment in 2012 was over seven times the number receiving treatment in 2005 , with an almost 1 million added in the last year alone . The
number of AIDS - related deaths in Sub-Saharan Africa in 2011 was 33 percent less than the number in 2005 . The number of new HIV infections in Sub-Saharan
Africa in 2011 was 25 percent less than the number in 2001 . Estimated prevalence in % of HIV among young adults ( 15 -- 49 ) per country as of 2011 . Malaria
is an endemic illness in Sub-Saharan Africa , where the majority of malaria cases and deaths worldwide occur . Routine immunization has been introduced in order
to prevent measles .
</item 7>

<item 8>
List of countries by HIV/AIDS adult prevalence rate
Over 10 % of all people living with HIV / AIDS reside within the region . Adult HIV prevalence exceeds 20 % in Swaziland , Botswana , and Lesotho , while an
additional six countries report adult HIV prevalence of at least 10 % . Outside Africa , the highest prevalence rate is found in the Bahamas ( 3.3 % ) . In
absolute numbers , South Africa ( 7.1 million ) , followed by Nigeria ( 3.2 million ) , and India ( 2.1 million ) had the highest HIV / AIDS number of cases by
the end of 2016 . While South Africa 's large population of HIV - positive people is attributable to its high disease prevalence ( 18.9 % , one of the highest
in the world ) , Nigeria 's is lower at 2.9 % , and India 's prevalence at 0.3 % is comparable to Canada and the UK , and is lower than the United States .
</item 8>

<item 9>
List of countries by HIV/AIDS adult prevalence rate
000 4,300 2016 Zimbabwe 13.50 % 1,300,000 30,000 2016 Zambia 12.40 % 1,200,000 21,000 2016 Mozambique 12.30 % 1,800,000 62,000 2016 Malawi 9.20 % 1,000,000
24,000 2016 Uganda 6.50 % 1,400,000 28,000 2016 Equatorial Guinea 6.20 % 35,000 - 2016 Kenya 5.40 % 1,600,000 36,000 2016 Tanzania 4.70 % 1,400,000 33,000 2016
Central African Republic 4.00 % 130,000 7,300 2016 Cameroon 3.80 % 560,000 29,000 2016 Gabon 3.60 % 48,000 1,500 2016 Bahamas , The 3.30 % 8,200 - 2016 Rwanda
3.10 % 220,000 3,
</item 9>

<item 10>
List of countries by HIV/AIDS adult prevalence rate
List of countries by HIV / AIDS adult prevalence rate - Wikipedia List of countries by HIV / AIDS adult prevalence rate Jump to : navigation , search This
article needs to be updated . Please update this article to reflect recent events or newly available information . ( October 2017 ) AIDS and HIV prevalence 2009
No data < 0.1 % 0.1 -- 0.5 % 0.5 -- 1 % 1 -- 5 % 5 -- 15 % 15 -- 50 % The human immunodeficiency virus ( HIV ) , which causes AIDS , varies in prevalence from
nation to nation . Listed here are the prevalence rates among adults in various countries , based on data from various sources , largely the CIA World Factbook
. As of 2015 , it is estimated that there are 36.7 million people worldwide infected with HIV . The HIV pandemic is most severe in South Africa . Over 10 % of
all people living with HIV / AIDS reside within the region .
</item 10>
=== END ITEM id=a3be46bbda174b90 ===

=== ITEM id=c1111e7e13275405 ===
Gold sentences and facts:
Sentence 0: Each state controls where it administers its bar exam . Because the MBE ( below ) is a standardized test , it must be administered on the same day across the country . That day occurs twice a year as the last Wednesday in July and the last Wednesday in February . Two states , Delaware and North Dakota , may administer their bar exams only once , in July , if they do not have enough applicants to merit a second sitting . North Dakota requires ten applicants in order to administer the February exam . Most bar exams are administered on consecutive days . Louisiana is the exception , with the Louisiana Bar Exam being a three - day examination on Monday , Wednesday and Friday . Also , Louisiana 's examination is the longest in the country in terms of examination time , with seven hours on Monday and Wednesday and seven and one half hours on Friday for a total of 21.5 hours of testing . Montana 's bar examination also occurs over a three - day period , with a total of 18 hours of testing . The bar exams in Delaware , Nevada , Ohio , South Carolina , and Texas are also three days long . The California bar exam moved to a two - day format beginning with the July , 2017 exam .
  Fact 0.0: Each state controls where it administers its bar exam.
  Fact 0.1: The MBE is a standardized test.
  Fact 0.2: The MBE must be administered on the same day across the country.
  Fact 0.3: The MBE day occurs twice a year.
  Fact 0.4: The MBE day is the last Wednesday in July and the last Wednesday in February.
  Fact 0.5: Delaware and North Dakota may administer their bar exams only once, in July.
  Fact 0.6: Delaware and North Dakota may do this if they do not have enough applicants to merit a second sitting.
  Fact 0.7: North Dakota requires ten applicants in order to administer the February exam.
  Fact 0.8: Most bar exams are administered on consecutive days.
  Fact 0.9: Louisiana is the exception to consecutive-day bar exams.
  Fact 0.10: The Louisiana Bar Exam is a three-day examination on Monday, Wednesday and Friday.
  Fact 0.11: Louisiana's examination is the longest in the country in terms of examination time.
  Fact 0.12: Louisiana's examination has seven hours on Monday and Wednesday.
  Fact 0.13: Louisiana's examination has seven and one half hours on Friday.
  Fact 0.14: Louisiana's examination totals 21.5 hours of testing.
  Fact 0.15: Montana's bar examination occurs over a three-day period.
  Fact 0.16: Montana's bar examination totals 18 hours of testing.
  Fact 0.17: The bar exams in Delaware, Nevada, Ohio, South Carolina, and Texas are three days long.
  Fact 0.18: The California bar exam moved to a two-day format beginning with the July 2017 exam.
Sentence 1: To refresh their memory on `` black - letter rules '' tested on the bar , most students engage in a regimen of study ( called `` bar review '' ) between graduating from law school and sitting for the bar . For bar review , most students in the United States attend a private bar review course which is provided by a third - party company and not their law school .
  Fact 1.0: Most students engage in a regimen of study called bar review.
  Fact 1.1: Bar review is done to refresh memory on black-letter rules tested on the bar.
  Fact 1.2: Bar review is done between graduating from law school and sitting for the bar.
  Fact 1.3: For bar review, most students in the United States attend a private bar review course.
  Fact 1.4: The private bar review course is provided by a third-party company.
  Fact 1.5: Most students in the United States do not take their bar review course from their law school.

Retrieved items:
<item 1>
Bar examination
Other states split the integrated bar membership and the admissions agency into different bodies within the judiciary ; in Texas , the Board of Law Examiners is
appointed by the Texas Supreme Court and is independent from the integrated State Bar of Texas . The bar examination in most U.S. states and territories is at
least two days long ( a few states have three - day exams ) and usually consists of : Essay questions : Essentially all jurisdictions administer several such
questions that test knowledge of general legal principles , and may also test knowledge of the state 's own law ( usually subjects such as wills , trusts and
community property , which always vary from one state to another ) . Some jurisdictions choose to use the Multistate Essay Examination ( MEE ) , drafted by the
NCBE since 1988 , for this purpose . Others may draft their own questions with this goal in mind , while some states both draft their own questions and use the
MEE . Some jurisdictions administer complicated questions that specifically test knowledge of that state 's law . Multistate standardized examinations ( below )
When exams occur ( edit ) Each state controls where it administers its bar exam .
</item 1>

<item 2>
Bar examination
A logbook signed by the judge on the bench has to certify their weekly attendance . By the end of the eighteenth month , they are eligible to apply to take the
Final Bar Exam by submitting their case summaries , the logbook and a research work pre-approved by the Bar . It is noteworthy , however , that during these
eighteen months , Trainees are eligible to have a limited practice of law under the supervision of their supervising Attorney . This practice does not include
Supreme Court eligible cases and certain criminal and civil cases . Candidates will be tested on Civil law , Civil Procedure , Criminal law , Criminal Procedure
, Commercial Law , Notary ( including rules pertaining Official Documents , Land & Real Estate registrations and regulations etc . ) . Each exam takes two days
, a day on oral examination in front of a judge or an attorney , and a day of essay examination , in which they will be tested on hypothetical cases submitted
to them . Successful applicants will be honoured with the title of `` First Degree Attorney '' , after they take the oath and can practice in all courts of the
country including the Supreme Court . Those who fail must redo the apprenticeship program in full or in part before taking the Final Bar Exam again .
</item 2>

<item 3>
Bar examination
Criminal law , employment law , constitution law , administrative law , and tax law . Civil procedure law , bankruptcy and business reorganization law , and the
system of the court of justice . And Criminal procedure law , human rights , and law on the evidence . Each part has 10 essay questions . The pass mark is 50 .
The parts 1 - 2 are usually taken in October and the rest are usually taken in March . One does not need to pass all four parts in one year . After passing all
the written exams , there is an oral exam . Around 10,000 bar students sit the exam each year . In 2013 , 1,231 students were called to the bar , 111 of whom
did it in only one year . Quite confusingly with international norms , students called to the bar are referred to as netibandit ( เนติบัณฑิต ) , which gets
translated into English as Barrister - at - Law . The Thai legal profession , however , is a fused one and those with lawyer licenses are able to practice both
as barristers and solicitors in the British / Commonwealth sense .
</item 3>

<item 4>
Bar examination
This exam is composed of three parts : Criminal Law , Criminal Procedural Law and Law of Criminal Enforcement Civil Law , Civil Procedural Law and Business Law
Constitutional Law , Administrative Law , Labor Law , Social Security Law and Law of the European Union . After passing these exams the candidate can practice
law as an attorney - at - law / barrister or as a secretary / judge at the court or as a prosecutor at the public prosecutor 's office or in - house legal
counsel or may operate individually at any field of law . Ireland ( edit ) The bar exams in Ireland are the preserve of the Honorable Society of King 's Inns ,
which runs a series of fourteen exams over ten weeks , from March to June each year , for those enrolled as students in its one - year Barrister - at - Law
degree course . These exams cover such skills as advocacy , research and opinion writing , consulting with clients , negotiation , drafting of legal documents
and knowledge of civil and criminal procedure . For those who fail to meet the requisite 50 % pass mark , repeats are held in the following August and September
.
</item 4>

<item 5>
Bar examination
Int'l L.J. 377 , 392 ( 2006 ) . This article contrasts American bar review courses against the 18 - month cram schools used in Japan , Germany , Korea , and
Taiwan , and argues that the short length of American bar review is due to the superior pedagogical methods of American law schools and the American tradition
of relatively easy access to the legal profession ( in comparison to most countries ) . Jump up ^ Wayne L. Anderson and Marilyn J. Headrick , The Legal
Profession : Is it for you ? ( Cincinnati : Thomson Executive Press , 1996 ) , 103 . Jump up ^ `` About NCBE '' . NCBE . Archived from the original on 2017 - 01
- 23 . Retrieved 2017 - 01 - 23 . Jump up ^ Bar Admissions statistics PDF ( accessed April 21 , 2009 ) Archived October 10 , 2008 , at the Wayback Machine .
Jump up ^ Cassens Weiss , Debra ( 2010 - 04 - 29 ) . `` With Missouri Move , Idea of Uniform Bar Exam Finally Gets Legs '' . ABA Journal .
</item 5>

<item 6>
Bar examination
The common law provinces all require prospective lawyers to complete a term of articles ( usually 10 months ) after graduation from law school during which they
work under the supervision of a qualified lawyer . The bar exams may be taken after graduation from law school , but before the commencement of articling , or
may be taken during or after articling is completed . Once the barristers ' and solicitors ' exams have been passed and the term of articles is successfully
completed , students may then be called the bar and admitted to the legal profession as lawyers ( barristers and solicitors ) . People 's Republic of China (
edit ) Main article : State Judicial Exam England and Wales ( edit ) Since the UK has a separated legal profession , Law graduates in England and Wales can take
examinations to qualify as a Barrister or a Solicitor by either undertaking the BPTC or the LPC respectively . These courses are the vocational part of the
training required under the Bar Association and The Law Society rules and are either undertaken on a full - time basis for one year or on a part - time basis
over two years .
</item 6>

<item 7>
Bar examination
The new law school system that began in 2009 allows only the graduates of a law school to apply for the bar exam . Malaysia ( edit ) Main article : Certificate
in Legal Practice ( Malaysia ) Philippines ( edit ) The Philippine Bar Examination is administered once every year on the four Sundays of November ( September
before 2011 ) . It covers eight areas of law : political law , labor law and social legislation , criminal law , civil law , commercial law , taxation law ,
remedial law , and legal ethics and practical exercises . Poland ( edit ) In Poland , the bar examination is taken after graduating from a law faculty at a
university . It allows a person to undertake practice , the duration of which varies depending on the specialization . After the practical period applicants
must pass the exam held by the Professional Chambers with assistance from some members of the Ministry of Justice . Singapore ( edit ) The Singapore Bar
Examination ( Part B) is administered once every year , usually over the course of four days . The exam is generally held on the last week of November , and is
administered by the Singapore Institute of Legal Education ( SILE ) .
</item 7>

<item 8>
Bar examination
The MEE and MPT , as uniform though not standardized tests , also must be administered on the same day across the country -- specifically the day before the MBE
. Preparation for the exam ( edit ) Most law schools teach students common law and how to analyze hypothetical fact patterns like a lawyer , but do not
specifically prepare law students for any particular bar exam . Only a minority of law schools offer bar preparation courses . To refresh their memory on ``
black - letter rules '' tested on the bar , most students engage in a regimen of study ( called `` bar review '' ) between graduating from law school and
sitting for the bar . For bar review , most students in the United States attend a private bar review course which is provided by a third - party company and
not their law school . Multistate standardized Examinations ( edit ) The National Conference of Bar Examiners ( NCBE ) is a U.S. based non-profit organization
that develops national ( `` multistate '' ) standardized tests for admission to the bar in individual states . The organization was founded in 1931 .
</item 8>

<item 9>
Bar examination
The exam is divided in two stages -- the first consists of 80 multiple choice questions covering all disciplines ( Ethics , Human Rights , Philosophy of Law ,
Constitutional Law , Administrative Law , Civil Law , Consumer Law , Civil Procedure Law , Criminal Law , Criminal Procedure Law , Labour Law and Labour
Procedure Law ) . The candidate must score at least 40 questions correctly to proceed to the second part of the exam , four essay questions and a drafting
project ( motion , opinion or claim document ) in Civil Law ( including Consumer Law ) , Labour Law , Criminal Law , Administrative Law , Constitutional Law ,
Corporate Law or Tax Law , and their respective procedures . The Bar examination can be taken on the graduation year . Success in the examination allows one to
practice in any court or jurisdiction of the country . Canada ( edit ) In Canada , admission to the bar is a matter of provincial or territorial jurisdiction .
All provinces except for Québec follow a common law tradition . Lawyers in every common law province are qualified as both barristers and solicitors , and must
pass a Barristers ' Examination and Solicitors ' Examination administered by the Law Society that governs the legal profession in their respective province or
territory .
</item 9>

<item 10>
Bar examination
After successfully completing these courses , which generally include various examinations and practical ability tests , graduates must secure either a Training
Contract ( for those who have completed the LPC ) or a Pupillage ( for those who have completed the BPTC ) . These are akin to Articling positions in other
jurisdictions and are the final Practical stage before being granted full admission to practice . The general timescale therefore to become fully qualified
after entering Law School can range between 6 -- 7 years ( assuming no repeats are required ) . However some controversy remains about the lack of Training
Contracts and Pupillages available to graduates even after having completed the LPC / BPTC . These courses can vary in cost anywhere from £ 9,000 to £ 17,000
and are generally undertaken by students on a private basis making them incur additional costs . The final debt in student fees alone after having completed the
academic and vocational training can range between £ 20,000 - £ 25,000 . This is set to increase to £ 40,000 - £ 50,000 for students entering law school in the
years 2012 due to the increased tuition fees for Law School itself .
</item 10>
=== END ITEM id=c1111e7e13275405 ===

=== ITEM id=a62e48bf7f345ddd ===
Gold sentences and facts:
Sentence 0: Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st Emmanuel Amunike Sporting CP 48 2nd George Weah Paris Saint - Germain 44 Rashidi Yekini Olympiacos 1995 1st George Weah Milan -- 2nd Emmanuel Amunike Sporting CP -- 3rd Daniel Amokachi Everton -- 1st Nwankwo Kanu Internazionale -- 2nd George Weah Milan -- 3rd Daniel Amokachi Beşiktaş -- 1st Victor Ikpeba Monaco 56 2nd Japhet N'Doram Monaco 40 3rd Taribo West Internazionale 35 1998 1st Mustapha Hadji Deportivo La Coruña 76 2nd Jay - Jay Okocha Paris Saint - Germain 74 3rd Sunday Oliseh Ajax 58 1999 1st Nwankwo Kanu Arsenal 46 2nd Samuel Kuffour Bayern Munich 44 3rd Ibrahima Bakayoko Marseille 42 2000 1st Patrick M'Boma Parma 123 2nd Lauren Mallorca 36 3rd Samuel Eto'o Mallorca 29 1st El Hadji Diouf Lens 93 2nd Samuel Kuffour Bayern Munich 66 3rd Samuel Eto'o Mallorca 34 2002 1st El Hadji Diouf Liverpool 93 2nd Papa Bouba Diop Lens 46 3rd Mido Ajax 42 2003 1st Samuel Eto'o Mallorca -- 2nd Didier Drogba Marseille -- 3rd Jay - Jay Okocha Bolton Wanderers -- 1st Samuel Eto'o Barcelona 116 2nd Didier Drogba Chelsea 90 3rd Jay - Jay Okocha Bolton Wanderers 68 2005 1st Samuel Eto'o Barcelona 108 2nd Didier Drogba Chelsea 106 3rd Michael Essien Chelsea 50 2006 1st Didier Drogba Chelsea 79 2nd Samuel Eto'o Barcelona 76 3rd Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74 2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea 43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311
  Fact 0.0: In 1992, Abedi Pele of Marseille ranked 1st with no points listed.
  Fact 0.1: In 1993, Rashidi Yekini of Vitória de Setúbal ranked 1st with no points listed.
  Fact 0.2: In 1994, Emmanuel Amunike of Sporting CP ranked 1st with 48 points.
  Fact 0.3: In 1994, George Weah of Paris Saint-Germain ranked 2nd with 44 points.
  Fact 0.4: In 1994, Rashidi Yekini of Olympiacos is listed with no rank shown.
  Fact 0.5: In 1995, George Weah of Milan ranked 1st with no points listed.
  Fact 0.6: In 1995, Emmanuel Amunike of Sporting CP ranked 2nd with no points listed.
  Fact 0.7: In 1995, Daniel Amokachi of Everton ranked 3rd with no points listed.
  Fact 0.8: In an entry with no year label, Nwankwo Kanu of Internazionale ranked 1st with no points listed.
  Fact 0.9: In an entry with no year label, George Weah of Milan ranked 2nd with no points listed.
  Fact 0.10: In an entry with no year label, Daniel Amokachi of Beşiktaş ranked 3rd with no points listed.
  Fact 0.11: In an entry with no year label, Victor Ikpeba of Monaco ranked 1st with 56 points.
  Fact 0.12: In an entry with no year label, Japhet N'Doram of Monaco ranked 2nd with 40 points.
  Fact 0.13: In an entry with no year label, Taribo West of Internazionale ranked 3rd with 35 points.
  Fact 0.14: In 1998, Mustapha Hadji of Deportivo La Coruña ranked 1st with 76 points.
  Fact 0.15: In 1998, Jay-Jay Okocha of Paris Saint-Germain ranked 2nd with 74 points.
  Fact 0.16: In 1998, Sunday Oliseh of Ajax ranked 3rd with 58 points.
  Fact 0.17: In 1999, Nwankwo Kanu of Arsenal ranked 1st with 46 points.
  Fact 0.18: In 1999, Samuel Kuffour of Bayern Munich ranked 2nd with 44 points.
  Fact 0.19: In 1999, Ibrahima Bakayoko of Marseille ranked 3rd with 42 points.
  Fact 0.20: In 2000, Patrick M'Boma of Parma ranked 1st with 123 points.
  Fact 0.21: In 2000, Lauren of Mallorca ranked 2nd with 36 points.
  Fact 0.22: In 2000, Samuel Eto'o of Mallorca ranked 3rd with 29 points.
  Fact 0.23: In an entry with no year label, El Hadji Diouf of Lens ranked 1st with 93 points.
  Fact 0.24: In an entry with no year label, Samuel Kuffour of Bayern Munich ranked 2nd with 66 points.
  Fact 0.25: In an entry with no year label, Samuel Eto'o of Mallorca ranked 3rd with 34 points.
  Fact 0.26: In 2002, El Hadji Diouf of Liverpool ranked 1st with 93 points.
  Fact 0.27: In 2002, Papa Bouba Diop of Lens ranked 2nd with 46 points.
  Fact 0.28: In 2002, Mido of Ajax ranked 3rd with 42 points.
  Fact 0.29: In 2003, Samuel Eto'o of Mallorca ranked 1st with no points listed.
  Fact 0.30: In 2003, Didier Drogba of Marseille ranked 2nd with no points listed.
  Fact 0.31: In 2003, Jay-Jay Okocha of Bolton Wanderers ranked 3rd with no points listed.
  Fact 0.32: In an entry with no year label, Samuel Eto'o of Barcelona ranked 1st with 116 points.
  Fact 0.33: In an entry with no year label, Didier Drogba of Chelsea ranked 2nd with 90 points.
  Fact 0.34: In an entry with no year label, Jay-Jay Okocha of Bolton Wanderers ranked 3rd with 68 points.
  Fact 0.35: In 2005, Samuel Eto'o of Barcelona ranked 1st with 108 points.
  Fact 0.36: In 2005, Didier Drogba of Chelsea ranked 2nd with 106 points.
  Fact 0.37: In 2005, Michael Essien of Chelsea ranked 3rd with 50 points.
  Fact 0.38: In 2006, Didier Drogba of Chelsea ranked 1st with 79 points.
  Fact 0.39: In 2006, Samuel Eto'o of Barcelona ranked 2nd with 76 points.
  Fact 0.40: In 2006, Michael Essien of Chelsea ranked 3rd with 36 points.
  Fact 0.41: In 2007, Frédéric Kanouté of Sevilla ranked 1st with no points listed.
  Fact 0.42: In 2007, Michael Essien of Chelsea ranked 2nd with no points listed.
  Fact 0.43: In 2007, Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.44: In 2008, Emmanuel Adebayor of Arsenal ranked 1st with 74 points.
  Fact 0.45: In 2008, Mohamed Aboutrika of Al-Ahly ranked 2nd with 53 points.
  Fact 0.46: In 2008, Michael Essien of Chelsea ranked 3rd with 36 points.
  Fact 0.47: In 2009, Didier Drogba of Chelsea ranked 1st with 92 points.
  Fact 0.48: In 2009, Samuel Eto'o of Internazionale ranked 2nd with 69 points.
  Fact 0.49: In 2009, Michael Essien of Chelsea ranked 3rd with 43 points.
  Fact 0.50: In an entry with no year label, Samuel Eto'o of Internazionale ranked 1st with no points listed.
  Fact 0.51: In an entry with no year label, Asamoah Gyan of Sunderland ranked 2nd with no points listed.
  Fact 0.52: In an entry with no year label, Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.53: In 2011, Yaya Touré of Manchester City ranked 1st with no points listed.
  Fact 0.54: In 2011, Seydou Keita of Barcelona ranked 2nd with no points listed.
  Fact 0.55: In 2011, André Ayew of Marseille ranked 3rd with no points listed.
  Fact 0.56: In 2012, Yaya Touré of Manchester City ranked 1st with no points listed.
  Fact 0.57: In 2012, Didier Drogba of Shanghai Shenhua ranked 2nd with no points listed.
  Fact 0.58: In 2012, Alex Song of Barcelona ranked 3rd with no points listed.
  Fact 0.59: In 2013, Yaya Touré of Manchester City ranked 1st with 373 points.
  Fact 0.60: In 2013, John Obi Mikel of Chelsea ranked 2nd with 265 points.
  Fact 0.61: In 2013, Didier Drogba of Galatasaray ranked 3rd with 236 points.
  Fact 0.62: In 2014, Yaya Touré of Manchester City ranked 1st with 175 points.
  Fact 0.63: In 2014, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 2nd with 120 points.
  Fact 0.64: In 2014, Vincent Enyeama of Lille ranked 3rd with 105 points.
  Fact 0.65: In 2015, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 1st with 143 points.
  Fact 0.66: In 2015, Yaya Touré of Manchester City ranked 2nd with 136 points.
  Fact 0.67: In 2015, André Ayew of Swansea City ranked 3rd with 112 points.
  Fact 0.68: In 2016, Riyad Mahrez of Leicester City ranked 1st with 361 points.
  Fact 0.69: In 2016, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 2nd with 313 points.
  Fact 0.70: In 2016, Sadio Mané of Liverpool ranked 3rd with 186 points.
  Fact 0.71: In 2017, Mohamed Salah of Liverpool ranked 1st with 625 points.
  Fact 0.72: In 2017, Sadio Mané of Liverpool ranked 2nd with 507 points.
  Fact 0.73: In 2017, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 3rd with 311 points.

Retrieved items:
<item 1>
African Footballer of the Year
Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2013 - 07 - 02 . Jump
up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2016 - 09 - 29 .
</item 1>

<item 2>
African Footballer of the Year
Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC Mulhouse -- -- Hearts of Oak -- -- FC Sion -- -- Aydınspor -- -- JS Kabylie -- --
Bolton Wanderers -- -- Everton FC -- -- Ajax Amsterdam -- -- Ismaily SC -- -- Zamalek SC -- AS Cannes -- -- Stade Lavallois -- -- Lille OSC -- -- Stade Rennais
-- -- Olympiacos -- -- Horoya AC -- -- PSV Eindhoven -- -- OC Agaza -- -- Beşiktaş -- -- Galatasaray -- -- Swansea City -- -- See also ( edit ) BBC African
Footballer of the Year African Women Footballer of the Year FIFA World Player of the Year CAF Awards References ( edit ) Jump up ^ `` African Player of the Year
1993 '' . Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year 1994 '' . Rsssf.com. 2001 - 01 - 05 .
</item 2>

<item 3>
African Footballer of the Year
Cannes 52 1992 1st Abedi Pele Marseille 198 2nd George Weah Paris Saint - Germain 161 3rd Tony Yeboah Eintracht Frankfurt 64 1993 1st Abedi Pele Marseille 119
2nd Tony Yeboah Eintracht Frankfurt 117 3rd Rashidi Yekini Vitória de Setúbal 104 1994 1st George Weah Paris Saint - Germain 148 2nd Emmanuel Amunike Sporting
CP 133 3rd Daniel Amokachi Everton 99 CAF award ( 1992 -- present ) ( edit ) Samuel Eto'o won the award four times . Yaya Touré won the award four consecutive
times from 2011 to 2014 . Pierre - Emerick Aubameyang was the first player from Gabon to win the award in 2015 . Mohamed Salah is the latest player to win the
award in 2017 . Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st
</item 3>

<item 4>
African Footballer of the Year
Keïta 1971 : Sunday 1972 : Souleymane 1973 : Bwanga 1974 : Moukila 1975 : Faras 1976 : Milla 1977 : Dhiab 1978 : Abdul Razak 1979 : N'Kono 1980 : Manga -
Onguéné 1981 : Belloumi 1982 : N'Kono 1983 : El Khatib 1984 : Abega 1985 : Timoumi 1986 : Zaki 1987 : Madjer 1988 : Bwalya 1989 : Weah 1990 : Milla 1991 : Abedi
Pele 1992 : Abedi Pele 1993 : Abedi Pele 1994 : Weah CAF award 1992 : Abedi Pele 1993 : Yekini 1994 : Amunike 1995 : Weah 1996 : Kanu 1997 : Ikpeba 1998 : Hadji
1999 : Kanu 2000 : M'Boma 2001 : Diouf 2002 :
</item 4>

<item 5>
African Footballer of the Year
It had already been replaced by an official award given out by the Confederation of African Football since 1992 . Year Rank Player Team Points 1970 1st Salif
Keïta Saint - Étienne 54 2nd Laurent Pokou ASEC Abidjan 28 Ali Abo Greisha Ismaily 1971 1st Ibrahim Sunday Asante Kotoko 29 2nd Robert Mensah Asante Kotoko 15
3rd François Ndoumbé Canon Yaoundé 13 1972 1st Chérif Souleymane Hafia 21 2nd Bwanga Tshimen TP Mazembe 16 3rd Petit Sory Hafia 14 1973 1st Bwanga Tshimen TP
Mazembe 49 2nd Kazadi Mwamba TP Mazembe 44 3rd Laurent Pokou ASEC Abidjan 41 1st Paul Moukila CARA Brazzaville 57 2nd Lobilo Boba Vita Club 32 3rd Hassan
Shehata Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd
</item 5>

<item 6>
African Footballer of the Year
Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74
2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea
43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita
Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya
Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014
</item 6>

<item 7>
African Footballer of the Year
African Footballer of the Year - wikipedia African Footballer of the Year Jump to : navigation , search The African Footballer of the Year award , presented to
the best African footballer each year , has been conferred by the Confederation of African Football ( CAF ) since 1992 . An earlier African Footballer of the
Year Golden Ball award was given out between 1970 and 1994 by France Football magazine . The changes resulted in parallel Golden Ball awards given out to Abedi
Pele and George Weah in 1993 and 1994 by the magazine although the CAF sponsored awards for those years were won respectively by Rashidi Yekini and Emmanuel
Amuneke , as well as two awards given to Abedi Pelé in 1992 . France Football discontinued the election from 1995 after the European Ballon d'Or -- also awarded
by the magazine -- had been opened to all players in the European leagues . On 2 February 2008 , Sevilla 's Frédéric Kanouté was named the 2007 African
Footballer of the Year .
</item 7>

<item 8>
African Footballer of the Year
Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd Roger Milla Tonnerre Yaoundé 24 Mamadou Aliou Kéïta Hafia 1976 1st Roger Milla Tonnerre Yaoundé 33 2nd Papa Camara
Hafia 32 3rd Ali Bencheikh MC Alger 27 1977 1st Tarak Dhiab Espérance Tunis 45 2nd Papa Camara Hafia 33 3rd Segun Odegbami Shooting Stars 29 1978 1st Karim
Abdul Razak Asante Kotoko 58 2nd Ali Bencheikh MC Alger 33 3rd Thomas N'Kono Canon Yaoundé 29 1979 1st Thomas N'Kono Canon Yaoundé 55 2nd Adolf Armah Hearts of
Oak 23 3rd Kerfalla Bangoura Horoya 15 1980 1st Jean Manga - Onguéné Canon Yaoundé 64 2nd Segun Odegbami Shooting Stars 41 3rd Théophile Abega Canon Yaoundé
</item 8>

<item 9>
African Footballer of the Year
DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- -- Chad -- -- Wins By Club ( edit ) Club Winners Runners - up Third places Olympique Marseille
Manchester City -- Chelsea 6 AS Monaco Barcelona Asante Kotoko -- Canon Yaoundé RCD Mallorca Inter Milan Liverpool Arsenal -- -- Paris SG -- Hafia Conacry TP
Mazembe -- Borussia Dortmund -- Sporting CP -- Al Ahly FC Porto Tonnerre Yaoundé -- RC Lens -- Milan -- Sporting CP -- GC Mascara -- Vitória de Setúbal --
Cercle Brugge -- -- CARA Brazzaville -- -- Leicester City -- -- AS Saint - Étienne -- -- Toulouse FC -- -- Parma -- -- FAR Rabat -- -- SCC Mohammédia -- -- JS
Saint - Pierroise -- -- Deportivo de La Coruña -- -- Espanyol -- -- Sevilla -- -- Espérance ST -- -- Bayern Munich -- -- MC Alger -- ASEC Mimosas -- Montpellier
-- Eintracht Frankfurt -- Shooting Stars -- Shanghai Shenhua -- -- Vita Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC
</item 9>

<item 10>
African Footballer of the Year
Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama
Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez
Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané
Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311 Wins By Country ( edit ) Nation Winners Runners - up Third places Cameroon 11 8 11 Ivory
Coast 8 6 7 Ghana 6 7 7 Nigeria 5 6 7 Morocco 0 Algeria 5 Liberia -- Egypt Mali -- Senegal Guinea DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- --
Chad -- -- Wins
</item 10>
=== END ITEM id=a62e48bf7f345ddd ===

=== ITEM id=e73e513d70bf9689 ===
Gold sentences and facts:
Sentence 0: State Pre-Prohibition ( prior to 1919 ) Post-Prohibition ( after 1933 ) 1970s / 26th Amendment ( adopted in 1971 ) 1980s / Drinking Age Act of 1984 21st century Alabama Pre 1881 : None 1881 : 21 ( none with consent of parent ) 21 1975 : Lowered to 19 1986 : Raised to 21 21 ( no one underage is allowed consumption Section 28 - 1 ) Alaska ? 21 1970 : Lowered to 19 1984 : Raised to 21 with grandfather clause ( if born before January 1 , 1965 ) 21 Arizona ? 21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and liquor 18 Raised to 21 in 1987 21 Connecticut ? 21 1972 : Lowered to 18 1982 : Raised to 19 1983 : Raised to 20 1985 : Raised to 21 21 ( exemptions : ( 1 ) a person over age eighteen who is an employee or permit holder under section 30 - 90a and who possesses alcoholic liquor in the course of such person 's employment or business , ( 2 ) a minor who possesses alcoholic liquor on the order of a practicing physician , or ( 3 ) a minor who possesses alcoholic liquor while accompanied by a parent , guardian or spouse of the minor , who has attained the age of twenty - one . Nothing in this subsection shall be construed to burden a person 's exercise of religion under section 3 of article first of the Constitution of the state in violation of subsection ( a ) of section 52 - 571b . Delaware ? 21 1972 : Lowered to 20 1984 : Raised to 21 21 District of Columbia ? 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 : Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 : Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to 19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ? Post-1934 : 21 21 21 21 Iowa ? 21 1972 : Lowered to 19 in 1972 ( Jul 1 ) 1973 : Lowered to 18 in 1973 ( Jul 1 ) 1978 : Raised to 19 in 1978 ( Jul 1 ) 1986 : Raised to 21 in 1986 ( Jul 1 ) with grandfather clause . 21 Kansas ? 1949 : 18 : for beer and wine with less than 4 % ABV 21 : for liquor 18 : for beer and wine with less than 4 % ABV 21 : for beer and wine with 4 % ABV or more 21 : for liquor 1985 : Raised to 21 21 ( However , anyone under 21 may consume cereal malt beverages with parental supervision on their own property ) Kentucky ? 1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 : De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision , raising to 21 three months later . Other exceptions still remain . 21 ( Exceptions to state law include : For an established religious purpose ; When a person under twenty - one years of age is accompanied by a parent , spouse , or legal guardian twenty - one years of age or older ; For medical purposes when purchased as an over the counter medication , or when prescribed or administered by a licensed physician , pharmacist , dentist , nurse , hospital , or medical institution ; In a private residence , which shall include a residential dwelling and up to twenty contiguous acres , on which the dwelling is located , owned by the same person who owns the dwelling ; The sale , handling , transport , or service in dispensing of any alcoholic beverage pursuant to lawful ownership of an establishment or to lawful employment of a person under twenty - one years of age by a duly licensed manufacturer , wholesaler , or retailer of beverage alcohol . ) Maine ? 21 1969 : Lowered to 20 1972 : Lowered to 18 1977 : Raised to 20 1985 : Raised to 21 21 Maryland ? 21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses , teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18 in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21 Mississippi ? 1966 : 18 : for beer and wine 21 : for liquor ( alcohol not legalized until 1966 ) 18 Raised to 21 in 1986 21 Missouri ? 1945 : 21 21 21 21 Montana ? 21 1971 : Lowered to 19 1973 : Lowered to 18 1979 : Raised to 19 1987 : Raised to 21 21 Nebraska ? 21 1969 : Lowered to 20 1972 : Lowered to 19 1980 : Raised to 20 1985 : Raised to 21 21 Nevada ? 1933 : 21 21 21 21 New Hampshire ? 21 1973 : Lowered to 18 1979 : Raised to 20 1985 : Raised to 21 21 New Jersey Before 1880 : None Post-1880 : 18 ( penalties only against businesses ) 21 1973 : Lowered to 18 1980 : Raised to 19 1983 : Raised to 21 21 New Mexico ? 1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 : for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ? 1935 : 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : beer 21 : wine and liquor 1988 : 21 21 ( However , anyone can drink with parents or legal - age spouse ) Oklahoma ? Men : 21 Women : 18 : 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1976 : Lowered to 18 ( for 3.2 % ABW beer ) for both sexes in 1976 ( Dec ) Last state to lower the drinking age . 1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to 20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 21 South Dakota ? 1934 : 21 ( all ) 1939 : Lowered to 18 : for 3.2 % beer Raised to 19 : for 3.2 % beer 1972 : 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1988 : Raised to 21 21 Tennessee Pre 1881 : None 1881 : 21 21 1964 : Lowered to 18 1979 : Raised to 19 1984 : Raised to 21 21 Texas Pre 1909 : 21 None ( with written permission of parent or guardian ) Post 1909 : 21 21 1973 : Lowered to 18 1981 : Raised to 19 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 Utah ? 1935 : 21 21 21 21 Vermont ? 21 1971 : Lowered to 18 1986 : Raised to 21 21 Virginia ? 21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer ) 1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all ) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all ) 18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales , remained 18 for on - premises ) . 1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 ( however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010
  Fact 0.0: The table lists each state's drinking age in five periods: pre-Prohibition (prior to 1919), post-Prohibition (after 1933), the 1970s (26th Amendment, adopted in 1971), the 1980s (Drinking Age Act of 1984), and the 21st century.
  Fact 0.1: Alabama had no drinking age before 1881.
  Fact 0.2: Alabama set its drinking age at 21 in 1881, with none with consent of parent.
  Fact 0.3: Alabama's drinking age was 21 after Prohibition.
  Fact 0.4: Alabama lowered its drinking age to 19 in 1975.
  Fact 0.5: Alabama raised its drinking age to 21 in 1986.
  Fact 0.6: Alabama's 21st-century drinking age is 21, and no one underage is allowed consumption under Section 28-1.
  Fact 0.7: Alaska's post-Prohibition drinking age was 21.
  Fact 0.8: Alaska lowered its drinking age to 19 in 1970.
  Fact 0.9: Alaska raised its drinking age to 21 in 1984, with a grandfather clause for those born before January 1, 1965.
  Fact 0.10: Alaska's 21st-century drinking age is 21.
  Fact 0.11: Arizona's post-Prohibition drinking age was 21.
  Fact 0.12: Arizona lowered its drinking age to 19 in 1972.
  Fact 0.13: Arizona raised its drinking age to 21 in 1985.
  Fact 0.14: Arizona's 21st-century drinking age is 21.
  Fact 0.15: Arkansas had no drinking age before 1925.
  Fact 0.16: Arkansas set its drinking age at 21 in 1925.
  Fact 0.17: Arkansas's drinking age stayed at 21 in the post-Prohibition, 1970s, 1980s and 21st-century periods.
  Fact 0.18: California was regulated by municipality or county before 1891, when the common age was 16.
  Fact 0.19: California set a statewide drinking age of 18 in 1891.
  Fact 0.20: California set its drinking age at 21 in 1933.
  Fact 0.21: California's drinking age stayed at 21 after 1933, except small amounts for religious ceremonies.
  Fact 0.22: Colorado had no drinking age before Prohibition.
  Fact 0.23: Colorado set its drinking age in 1945 at 18 for beer and 21 for wine and liquor.
  Fact 0.24: Colorado's drinking age was 18 in the 1970s.
  Fact 0.25: Colorado raised its drinking age to 21 in 1987.
  Fact 0.26: Colorado's 21st-century drinking age is 21.
  Fact 0.27: Connecticut's post-Prohibition drinking age was 21.
  Fact 0.28: Connecticut lowered its drinking age to 18 in 1972.
  Fact 0.29: Connecticut raised its drinking age to 19 in 1982.
  Fact 0.30: Connecticut raised its drinking age to 20 in 1983.
  Fact 0.31: Connecticut raised its drinking age to 21 in 1985.
  Fact 0.32: Connecticut's 21st-century drinking age is 21.
  Fact 0.33: Connecticut exempts a person over 18 who is an employee or permit holder under section 30-90a and who possesses alcoholic liquor in the course of such employment or business.
  Fact 0.34: Connecticut exempts a minor who possesses alcoholic liquor on the order of a practicing physician.
  Fact 0.35: Connecticut exempts a minor who possesses alcoholic liquor while accompanied by a parent, guardian or spouse of the minor who has attained the age of 21.
  Fact 0.36: Connecticut's age limit does not burden a person's exercise of religion under section 3 of article first of the state Constitution in violation of section 52-571b(a).
  Fact 0.37: Delaware's post-Prohibition drinking age was 21.
  Fact 0.38: Delaware lowered its drinking age to 20 in 1972.
  Fact 0.39: Delaware raised its drinking age to 21 in 1984.
  Fact 0.40: Delaware's 21st-century drinking age is 21.
  Fact 0.41: The District of Columbia set its drinking age at 18 for beer and wine and 21 for liquor.
  Fact 0.42: The District of Columbia's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.43: The District of Columbia raised its drinking age to 21 in 1986, with a grandfather clause.
  Fact 0.44: The District of Columbia's 21st-century drinking age is 21.
  Fact 0.45: Florida's drinking age was 18 in the 1970s.
  Fact 0.46: Florida raised its drinking age to 19 in 1980.
  Fact 0.47: Florida raised its drinking age to 21 in 1985, with a grandfather clause.
  Fact 0.48: Florida's 21st-century drinking age is 21.
  Fact 0.49: Georgia's post-Prohibition drinking age was 21.
  Fact 0.50: Georgia lowered its drinking age to 18 in 1972.
  Fact 0.51: Georgia raised its drinking age to 19 in 1984.
  Fact 0.52: Georgia raised its drinking age to 20 in 1985.
  Fact 0.53: Georgia raised its drinking age to 21 in 1986.
  Fact 0.54: Georgia's 21st-century drinking age is 21.
  Fact 0.55: Hawaii's post-Prohibition drinking age was 21.
  Fact 0.56: Hawaii lowered its drinking age to 18 in 1972.
  Fact 0.57: Hawaii raised its drinking age to 21 in 1986.
  Fact 0.58: Hawaii's 21st-century drinking age is 21.
  Fact 0.59: Idaho's post-Prohibition drinking age was 20 for beer and 21 for wine and liquor.
  Fact 0.60: Idaho lowered its drinking age to 19 for all alcohol in 1972.
  Fact 0.61: Idaho raised its drinking age to 21 on April 11, 1987, with a grandfather clause.
  Fact 0.62: Idaho's 21st-century drinking age is 21.
  Fact 0.63: Illinois had a drinking age equal to the age of majority in 1872, with none with written consent of parent.
  Fact 0.64: Illinois's drinking age was the age of majority before 1961.
  Fact 0.65: Illinois raised its drinking age to 21 in 1961.
  Fact 0.66: Illinois lowered its drinking age to 19 for beer and wine in 1973, and kept 21 for liquor.
  Fact 0.67: Illinois raised its drinking age to 21 in 1980.
  Fact 0.68: Illinois's 21st-century drinking age is 21.
  Fact 0.69: Indiana's drinking age has been 21 since 1934.
  Fact 0.70: Indiana's drinking age was 21 in the 1970s, 1980s and 21st century.
  Fact 0.71: Iowa's post-Prohibition drinking age was 21.
  Fact 0.72: Iowa lowered its drinking age to 19 on July 1, 1972.
  Fact 0.73: Iowa lowered its drinking age to 18 on July 1, 1973.
  Fact 0.74: Iowa raised its drinking age to 19 on July 1, 1978.
  Fact 0.75: Iowa raised its drinking age to 21 on July 1, 1986, with a grandfather clause.
  Fact 0.76: Iowa's 21st-century drinking age is 21.
  Fact 0.77: Kansas set its post-Prohibition drinking age in 1949 at 18 for beer and wine under 4% ABV and 21 for liquor.
  Fact 0.78: Kansas's 1970s drinking age was 18 for beer and wine under 4% ABV, and 21 for beer and wine at 4% ABV or more and for liquor.
  Fact 0.79: Kansas raised its drinking age to 21 in 1985.
  Fact 0.80: Kansas's 21st-century drinking age is 21, but anyone under 21 may consume cereal malt beverages with parental supervision on their own property.
  Fact 0.81: Kentucky's drinking age was 21 from 1938 onward.
  Fact 0.82: Louisiana's drinking age was 18 as of 1948.
  Fact 0.83: Louisiana's drinking age was 18 in the 1970s.
  Fact 0.84: Louisiana's drinking age was raised to 21 de jure in 1987, but the de facto age stayed at 18 until 1995 because of a sale loophole.
  Fact 0.85: Louisiana's de facto drinking age was raised to 21 in 1995 when the loophole was closed.
  Fact 0.86: In 1996 the Louisiana Supreme Court briefly lowered the drinking age to 18, until it reversed its decision and raised it to 21 three months later.
  Fact 0.87: Louisiana's 21st-century drinking age is 21, with exceptions that remain.
  Fact 0.88: Louisiana's drinking age exception covers an established religious purpose.
  Fact 0.89: Louisiana's drinking age exception covers a person under 21 accompanied by a parent, spouse, or legal guardian who is 21 or older.
  Fact 0.90: Louisiana's drinking age exception covers medical purposes, when purchased as an over-the-counter medication or when prescribed or administered by a licensed physician, pharmacist, dentist, nurse, hospital, or medical institution.
  Fact 0.91: Louisiana's drinking age exception covers a private residence, including a residential dwelling and up to twenty contiguous acres on which the dwelling is located, owned by the same person who owns the dwelling.
  Fact 0.92: Louisiana's drinking age exception covers the sale, handling, transport, or service in dispensing of alcoholic beverages under lawful ownership of an establishment or lawful employment of a person under 21 by a duly licensed manufacturer, wholesaler, or retailer of beverage alcohol.
  Fact 0.93: Maine's post-Prohibition drinking age was 21.
  Fact 0.94: Maine lowered its drinking age to 20 in 1969.
  Fact 0.95: Maine lowered its drinking age to 18 in 1972.
  Fact 0.96: Maine raised its drinking age to 20 in 1977.
  Fact 0.97: Maine raised its drinking age to 21 in 1985.
  Fact 0.98: Maine's 21st-century drinking age is 21.
  Fact 0.99: Maryland's post-Prohibition drinking age was 21.
  Fact 0.100: Maryland lowered its drinking age to 18 for beer and wine in 1974, and kept 21 for liquor.
  Fact 0.101: Maryland raised its drinking age to 21 in 1982, with a grandfather clause.
  Fact 0.102: Maryland's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.103: Massachusetts's post-Prohibition drinking age was 21.
  Fact 0.104: Massachusetts lowered its drinking age to 18 in 1973.
  Fact 0.105: Massachusetts raised its drinking age to 20 on April 16, 1979.
  Fact 0.106: Massachusetts raised its drinking age to 21 in 1985.
  Fact 0.107: Massachusetts's 21st-century drinking age is 21.
  Fact 0.108: Michigan's post-Prohibition drinking age was 21.
  Fact 0.109: Michigan lowered its drinking age to 18 on January 1, 1972.
  Fact 0.110: Michigan raised its drinking age to 19 on December 3, 1978.
  Fact 0.111: Michigan raised its drinking age to 21 on December 21, 1978, 18 days later.
  Fact 0.112: Michigan was the first state to raise its age to 21 after lowering it.
  Fact 0.113: Michigan's 21st-century drinking age is 21.
  Fact 0.114: Minnesota's post-Prohibition drinking age was 21.
  Fact 0.115: Minnesota lowered its drinking age to 18 on June 1, 1973.
  Fact 0.116: Minnesota raised its drinking age to 19 on September 1, 1976.
  Fact 0.117: Minnesota was the first state to raise its drinking age after lowering it.
  Fact 0.118: Minnesota raised its drinking age to 21 in 1986.
  Fact 0.119: Minnesota's 21st-century drinking age is 21.
  Fact 0.120: Mississippi set a drinking age of 18 for beer and wine and 21 for liquor in 1966, when alcohol was legalized.
  Fact 0.121: Mississippi's drinking age was 18 in the 1970s.
  Fact 0.122: Mississippi raised its drinking age to 21 in 1986.
  Fact 0.123: Mississippi's 21st-century drinking age is 21.
  Fact 0.124: Missouri's drinking age was 21 from 1945 onward.
  Fact 0.125: Montana's post-Prohibition drinking age was 21.
  Fact 0.126: Montana lowered its drinking age to 19 in 1971.
  Fact 0.127: Montana lowered its drinking age to 18 in 1973.
  Fact 0.128: Montana raised its drinking age to 19 in 1979.
  Fact 0.129: Montana raised its drinking age to 21 in 1987.
  Fact 0.130: Montana's 21st-century drinking age is 21.
  Fact 0.131: Nebraska's post-Prohibition drinking age was 21.
  Fact 0.132: Nebraska lowered its drinking age to 20 in 1969.
  Fact 0.133: Nebraska lowered its drinking age to 19 in 1972.
  Fact 0.134: Nebraska raised its drinking age to 20 in 1980.
  Fact 0.135: Nebraska raised its drinking age to 21 in 1985.
  Fact 0.136: Nebraska's 21st-century drinking age is 21.
  Fact 0.137: Nevada's drinking age was 21 from 1933 onward.
  Fact 0.138: New Hampshire's post-Prohibition drinking age was 21.
  Fact 0.139: New Hampshire lowered its drinking age to 18 in 1973.
  Fact 0.140: New Hampshire raised its drinking age to 20 in 1979.
  Fact 0.141: New Hampshire raised its drinking age to 21 in 1985.
  Fact 0.142: New Hampshire's 21st-century drinking age is 21.
  Fact 0.143: New Jersey had no drinking age before 1880.
  Fact 0.144: New Jersey set a drinking age of 18 after 1880, with penalties only against businesses.
  Fact 0.145: New Jersey's post-Prohibition drinking age was 21.
  Fact 0.146: New Jersey lowered its drinking age to 18 in 1973.
  Fact 0.147: New Jersey raised its drinking age to 19 in 1980.
  Fact 0.148: New Jersey raised its drinking age to 21 in 1983.
  Fact 0.149: New Jersey's 21st-century drinking age is 21.
  Fact 0.150: New Mexico's drinking age was 21 from 1934 onward.
  Fact 0.151: New York's pre-Prohibition drinking age was 21.
  Fact 0.152: New York's post-Prohibition drinking age was 18.
  Fact 0.153: New York's drinking age was 18 in the 1970s.
  Fact 0.154: New York raised its drinking age to 19 in December 1982.
  Fact 0.155: New York raised its drinking age to 21 in December 1985.
  Fact 0.156: New York's 21st-century drinking age is 21.
  Fact 0.157: North Carolina's post-Prohibition drinking age was 18.
  Fact 0.158: North Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.159: North Carolina's 1983 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.160: North Carolina raised its drinking age to 21 on September 1, 1986.
  Fact 0.161: North Carolina's 21st-century drinking age is 21.
  Fact 0.162: North Dakota's drinking age was 21 from 1936 onward.
  Fact 0.163: Ohio's pre-Prohibition drinking age is unknown.
  Fact 0.164: Ohio set its post-Prohibition drinking age in 1935 at 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.165: Ohio's 1970s drinking age was 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.166: In 1982 Ohio set the age at 19 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.167: A second 1982 Ohio entry set the age at 19 for beer and 21 for wine and liquor.
  Fact 0.168: Ohio raised its drinking age to 21 in 1988.
  Fact 0.169: Ohio's 21st-century drinking age is 21, and anyone can drink with parents or a legal-age spouse.
  Fact 0.170: Oklahoma's drinking age was 21 for men and 18 for women for 3.2% ABW beer, and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.171: Oklahoma lowered its drinking age to 18 for 3.2% ABW beer for both sexes in December 1976.
  Fact 0.172: Oklahoma was the last state to lower the drinking age.
  Fact 0.173: Oklahoma raised its drinking age to 21 in 1983.
  Fact 0.174: Oklahoma's 21st-century drinking age is 21.
  Fact 0.175: Oregon's drinking age was 21 from 1933 onward.
  Fact 0.176: Pennsylvania's drinking age was 21 from 1935 onward.
  Fact 0.177: Rhode Island's post-Prohibition drinking age was 21.
  Fact 0.178: Rhode Island lowered its drinking age to 18 in 1970.
  Fact 0.179: Rhode Island raised its drinking age to 19 in 1980.
  Fact 0.180: Rhode Island raised its drinking age to 20 in 1981.
  Fact 0.181: Rhode Island raised its drinking age to 21 in 1984.
  Fact 0.182: Rhode Island's 21st-century drinking age is 21.
  Fact 0.183: South Carolina's pre-Prohibition drinking age was 21.
  Fact 0.184: South Carolina set its post-Prohibition drinking age in 1935 at 18 for beer and wine and 21 for liquor.
  Fact 0.185: South Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.186: South Carolina's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.187: South Carolina raised its drinking age to 21 in 1986.
  Fact 0.188: South Carolina's 21st-century drinking age is 21.
  Fact 0.189: South Dakota's drinking age was 21 for all alcohol in 1934.
  Fact 0.190: South Dakota lowered its drinking age to 18 for 3.2% beer in 1939 and raised it to 19 for 3.2% beer.
  Fact 0.191: South Dakota's 1972 drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.192: South Dakota's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.193: South Dakota raised its drinking age to 21 in 1988.
  Fact 0.194: South Dakota's 21st-century drinking age is 21.
  Fact 0.195: Tennessee had no drinking age before 1881.
  Fact 0.196: Tennessee set its drinking age at 21 in 1881.
  Fact 0.197: Tennessee's post-Prohibition drinking age was 21.
  Fact 0.198: Tennessee lowered its drinking age to 18 in 1964.
  Fact 0.199: Tennessee raised its drinking age to 19 in 1979.
  Fact 0.200: Tennessee raised its drinking age to 21 in 1984.
  Fact 0.201: Tennessee's 21st-century drinking age is 21.
  Fact 0.202: Texas's drinking age was 21 before 1909, with none with written permission of parent or guardian.
  Fact 0.203: Texas's drinking age was 21 after 1909 and after Prohibition.
  Fact 0.204: Texas lowered its drinking age to 18 in 1973.
  Fact 0.205: Texas raised its drinking age to 19 in 1981.
  Fact 0.206: Texas raised its drinking age to 21 on September 1, 1986.
  Fact 0.207: Texas's 21st-century drinking age is 21.
  Fact 0.208: Utah's drinking age was 21 from 1935 onward.
  Fact 0.209: Vermont's post-Prohibition drinking age was 21.
  Fact 0.210: Vermont lowered its drinking age to 18 in 1971.
  Fact 0.211: Vermont raised its drinking age to 21 in 1986.
  Fact 0.212: Vermont's 21st-century drinking age is 21.
  Fact 0.213: Virginia's post-Prohibition drinking age was 21.
  Fact 0.214: Virginia lowered its drinking age to 18 for beer and wine in 1974, while 21 for liquor remained.
  Fact 0.215: Virginia raised its drinking age to 19 for off-premises consumption in 1981.
  Fact 0.216: Virginia raised its drinking age to 19 for all beer in 1983.
  Fact 0.217: Virginia raised its drinking age to 21 in 1985.
  Fact 0.218: Virginia's 21st-century drinking age is 21.
  Fact 0.219: Washington's drinking age has been the age of majority since 1877.
  Fact 0.220: Washington's drinking age was 21 from after Prohibition onward.
  Fact 0.221: West Virginia's post-Prohibition drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.222: West Virginia lowered its drinking age to 18 for all alcohol in 1972.
  Fact 0.223: West Virginia raised its drinking age to 19 in 1983, and to 21 for non-residents.
  Fact 0.224: West Virginia raised its drinking age to 21 in 1986.
  Fact 0.225: West Virginia's 21st-century drinking age is 21.
  Fact 0.226: Wisconsin's drinking age was 18 for wine and liquor, with none for beer, from 1839 to 1866.
  Fact 0.227: Wisconsin set its drinking age at 21 for all alcohol after 1866.
  Fact 0.228: Wisconsin's post-Prohibition drinking age was 18 for beer and 21 for wine and spirits.
  Fact 0.229: In 1957 Wisconsin set a 21 limit for residents of bordering states with an age limit of 21.
  Fact 0.230: In 1963 Wisconsin raised its age to 21 for off-premises beer sales, while it remained 18 for on-premises.
  Fact 0.231: Wisconsin lowered its drinking age to 18 for all alcohol in March 1972.
  Fact 0.232: Wisconsin lifted the border state restriction in 1977.
  Fact 0.233: Wisconsin raised its drinking age to 19 in 1984.
  Fact 0.234: Wisconsin raised its drinking age to 21 on September 1, 1986.
  Fact 0.235: Wisconsin's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.236: Wyoming's post-Prohibition drinking age was 21.
  Fact 0.237: Wyoming lowered its drinking age to 19 in 1973.
  Fact 0.238: Wyoming raised its drinking age to 21 on July 1, 1988.
  Fact 0.239: Wyoming was the last state to raise its de jure drinking age to 21.
  Fact 0.240: Wyoming's 21st-century drinking age is 21.
  Fact 0.241: American Samoa's listed drinking age is 21.
  Fact 0.242: The Northern Mariana Islands' listed drinking age is 21.
  Fact 0.243: Puerto Rico's post-Prohibition drinking age was 18.
  Fact 0.244: Puerto Rico's 1970s drinking age was 18.
  Fact 0.245: Puerto Rico kept its drinking age at 18 despite a 10% highway funding penalty under the Drinking Age Act.
  Fact 0.246: Puerto Rico's 21st-century drinking age is 18.
  Fact 0.247: The Virgin Islands' drinking age was 18 in the 1980s and in the 21st century.
  Fact 0.248: Guam's listed drinking age includes 21 and 18.
  Fact 0.249: Guam's 1980s and 21st-century drinking age was 18.
  Fact 0.250: Guam raised its drinking age to 21 in 2010.
Sentence 1: From 1976 to 1983 , several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 % of their federal highway funds . By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
  Fact 1.0: From 1976 to 1983, several states voluntarily raised their purchase ages to 19 (or, less commonly, 20 or 21).
  Fact 1.1: The voluntary purchase-age increases from 1976 to 1983 were in part to combat drunk driving fatalities.
  Fact 1.2: In 1984 Congress passed the National Minimum Drinking Age Act.
  Fact 1.3: The National Minimum Drinking Age Act required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10% of their federal highway funds.
  Fact 1.4: By mid-1988, all 50 states and the District of Columbia had raised their purchase ages to 21.
  Fact 1.5: Puerto Rico, Guam, and the Virgin Islands had not raised their purchase ages to 21 by mid-1988.
  Fact 1.6: South Dakota and Wyoming were the final two states to comply with the age 21 mandate.
  Fact 1.7: The current drinking age of 21 remains a point of contention among many Americans.
  Fact 1.8: The current drinking age of 21 is higher than the age of majority, which is 18 in most states.
  Fact 1.9: The current drinking age of 21 is higher than the drinking ages of most other countries.
  Fact 1.10: The National Minimum Drinking Age Act is seen as a congressional sidestep of the tenth amendment.
  Fact 1.11: Debates over the drinking age have not been highly publicized.
  Fact 1.12: A few states have proposed legislation to lower their drinking age.
  Fact 1.13: Guam raised its drinking age to 21 in July 2010.

Retrieved items:
<item 1>
U.S. history of alcohol minimum purchase age by state
Minimum legal drinking age as of 1983 ( one year before the Drinking Age Act of 1984 was passed ) : Minimum age is 21 Minimum age is 20 Minimum age is 19 and 21
Minimum age is 19 Minimum age is 18 and 21 Minimum age is 18 The alcohol laws of the United States regarding minimum age for purchase have changed over time .
The history is given in the table below . Unless otherwise noted , if different alcohol categories have different minimum purchase ages , the age listed below
is set at the lowest age given ( e.g. if the purchase age is 18 for beer and 21 for wine or spirits , as was the case in several states , the age in the table
will read as `` 18 '' , not `` 21 '' ) . In addition , the purchase age is not necessarily the same as the minimum age for consumption of alcoholic beverages ,
although they have often been the same . As one can see in the table below , there has been much volatility in the states ' drinking ages since the repeal of
Prohibition in 1933 .
</item 1>

<item 2>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer )
1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all
) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all )
18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales ,
remained 18 for on - premises ) .
</item 2>

<item 3>
U.S. history of alcohol minimum purchase age by state
18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 :
Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 :
Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with
grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to
19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ?
</item 3>

<item 4>
U.S. history of alcohol minimum purchase age by state
Not only did it still allow 18 - to 20 - year - olds to consume in private , it contained a major loophole allowing bars and stores to sell alcohol to 18 - to
20 - year - olds without penalty ( despite purchase being technically illegal ) which meant that the de facto age was still 18 . In other words , the drinking
age was 21 only on paper . This loophole was closed in 1995 , but in 1996 the Louisiana Supreme Court declared a drinking age of 21 unconstitutional . That
briefly lowered the de jure purchase age to 18 , causing an uproar which prompted the Louisiana Supreme Court to reverse its decision , raising the age to 21
three months later . Other exceptions still remain to this day , including drinking in a private residence , and Louisiana still has some of the most liberal
general alcohol laws of any state . Some states were `` dry '' well before national Prohibition was enacted in 1919 , in some cases since achieving statehood .
Also , some states did not become fully `` wet '' until several years after the repeal of Prohibition in 1933 ( e.g.
</item 4>

<item 5>
U.S. history of alcohol minimum purchase age by state
Shortly after the ratification of the 21st amendment in December , most states set their purchase ages at 21 since that was the Voting age at the time . Most of
these limits remained constant until the early 1970s . From 1969 to 1976 , some 30 states lowered their purchase ages , generally to 18 . This was primarily
because the voting age was lowered from 21 to 18 in 1971 with the 26th amendment . A lot of states started to lower their minimum drinking age in response ,
most of this occurring in 1972 or 1973 . Twelve states kept their purchase ages at 21 since repeal of Prohibition and never changed them . From 1976 to 1983 ,
several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress
passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 %
of their federal highway funds .
</item 5>

<item 6>
U.S. history of alcohol minimum purchase age by state
1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 :
De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision ,
raising to 21 three months later . Other exceptions still remain .
</item 6>

<item 7>
U.S. history of alcohol minimum purchase age by state
1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 (
however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to
raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway
funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010 Additional Notes ( edit ) Contrary to popular
belief , since the National Minimum Drinking Age Act of 1984 , not all states specifically prohibit minors ' and young adults ' consumption of alcohol in
private settings .
</item 7>

<item 8>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses ,
teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18
in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was
lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21
Mississippi ?
</item 8>

<item 9>
U.S. history of alcohol minimum purchase age by state
By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see
Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point
of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other
countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly
publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
</item 9>

<item 10>
U.S. history of alcohol minimum purchase age by state
21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common
age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and
liquor 18 Raised to 21 in 1987 21 Connecticut ?
</item 10>
=== END ITEM id=e73e513d70bf9689 ===
