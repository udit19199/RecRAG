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

=== ITEM id=51ad445b29347956 ===
Gold sentences and facts:
Sentence 0: The Compromise of 1877 was an informal , unwritten deal that settled the intensely disputed 1876 U.S. presidential election . It resulted in the United States federal government pulling the last troops out of the South , and formally ended the Reconstruction Era . Through the Compromise , Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden on the understanding that Hayes would remove the federal troops whose support was essential for the survival of Republican state governments in South Carolina , Florida and Louisiana . The compromise involved Democrats who controlled the House of Representatives allowing the decision of the Electoral Commission to take effect . The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida . As president , Hayes removed the remaining troops from South Carolina and Louisiana . As soon as the troops left , many white Republicans also left , and the `` Redeemer '' Democrats took control . They already dominated other state governments in the South . What was exactly agreed is somewhat contested as the documentation is insufficient .
  Fact 0.0: The Compromise of 1877 was an informal, unwritten deal.
  Fact 0.1: The Compromise of 1877 settled the intensely disputed 1876 U.S. presidential election.
  Fact 0.2: The Compromise resulted in the United States federal government pulling the last troops out of the South.
  Fact 0.3: The Compromise formally ended the Reconstruction Era.
  Fact 0.4: Through the Compromise, Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden.
  Fact 0.5: The White House was awarded to Hayes on the understanding that Hayes would remove the federal troops.
  Fact 0.6: The federal troops' support was essential for the survival of Republican state governments in South Carolina, Florida and Louisiana.
  Fact 0.7: The compromise involved Democrats who controlled the House of Representatives.
  Fact 0.8: The compromise involved the Democrats allowing the decision of the Electoral Commission to take effect.
  Fact 0.9: The outgoing president, Republican Ulysses S. Grant, removed the soldiers from Florida.
  Fact 0.10: As president, Rutherford B. Hayes removed the remaining troops from South Carolina.
  Fact 0.11: As president, Rutherford B. Hayes removed the remaining troops from Louisiana.
  Fact 0.12: When the troops left, many white Republicans also left.
  Fact 0.13: When the troops left, the 'Redeemer' Democrats took control.
  Fact 0.14: The 'Redeemer' Democrats already dominated other state governments in the South.
  Fact 0.15: What was exactly agreed in the Compromise of 1877 is somewhat contested.
  Fact 0.16: The documentation of the Compromise of 1877 is insufficient.

Retrieved items:
<item 1>
Compromise of 1877
The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida . As president , Hayes removed the remaining troops from South Carolina
and Louisiana . As soon as the troops left , many white Republicans also left , and the `` Redeemer '' Democrats took control . They already dominated other
state governments in the South . What was exactly agreed is somewhat contested as the documentation is insufficient . Black Republicans felt betrayed as they
lost power and were subject to discrimination and harassment to suppress their voting . by 1905 most black men were effectively disenfranchised by state
legislatures in every southern state . Contents ( hide ) 1 Terms of compromise 2 Results 3 Interpretations 4 References 5 Further reading 6 External links Terms
of compromise ( edit ) The compromise essentially stated that Southern Democrats would acknowledge Hayes as president , but only on the understanding that
Republicans would meet certain demands . The following elements are generally said to be the points of the compromise : The removal of all U.S. military forces
from the former Confederate states . At the time , U.S.
</item 1>

<item 2>
Compromise of 1877
Compromise of 1877 - wikipedia Compromise of 1877 Jump to : navigation , search A political cartoon by Joseph Keppler depicts Roscoe Conkling as a character
Mephistopheles ( the Devil ) while Rutherford B. Hayes strolls off with the prize of the `` Solid South '' depicted as a woman . The caption quotes Goethe : ``
Unto that Power he doth belong Which only doeth Right while ever willing Wrong . '' The Compromise of 1877 was an informal , unwritten deal that settled the
intensely disputed 1876 U.S. presidential election . It resulted in the United States federal government pulling the last troops out of the South , and formally
ended the Reconstruction Era . Through the Compromise , Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden on the
understanding that Hayes would remove the federal troops whose support was essential for the survival of Republican state governments in South Carolina ,
Florida and Louisiana . The compromise involved Democrats who controlled the House of Representatives allowing the decision of the Electoral Commission to take
effect . The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida .
</item 2>

<item 3>
Compromise of 1877
References ( edit ) Jump up ^ Michael Les Benedict , `` Southern Democrats in the Crisis of 1876 - 1877 : A Reconsideration of Reunion and Reaction . '' Journal
of Southern History ( 1980 ) : 489 - 524 . in JSTOR Jump up ^ Jones , Stephen A. ; Freedman , Eric ( 2011 ) . Presidents and Black America . CQ Press . p. 218 .
ISBN 9781608710089 . In an eleventh - hour compromise between party leaders - considered the `` Great Betrayal '' by many blacks and southern Republicans ...
Jump up ^ Woodward , C. Vann ( 1966 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Boston : Little , Brown and Company . pp.
169 -- 171 . Jump up ^ Donald Richard Deskins ; Hanes Walton ; Sherman C. Puckett ( 2010 ) . Presidential Elections , 1789 - 2008 : County , State , and
National Mapping of Election Data . U of Michigan Press . p. 211 .
</item 3>

<item 4>
Compromise of 1877
( Oxford University Press , 1982 ) pp. 417 -- 50 . Downs , Gregory P. ( 2012 ) . `` The Mexicanization of American Politics : The United States ' Transnational
Path from Civil War to Stabilization '' . The American Historical Review . 117 ( 2 ) : 387 -- 409 . doi : 10.1086 / ahr. 117.2. 387 . Frantz , Edward O . The
Door of Hope : Republican Presidents and the First Southern Strategy , 1877 -- 1933 ( University Press of Florida . 2011 ) Peskin , Allan ( 1973 ) . `` Was
There a Compromise of 1877 '' . The Journal of American History . 60 ( 1 ) : 63 -- 75 . doi : 10.2307 / 2936329 . ISSN 1936 - 0967 . JSTOR 2936329 -- via JSTOR
. ( Registration required ( help ) ) . Polakoff , Keith Ian .
</item 4>

<item 5>
Compromise of 1877
He tightened military security , and nobody marched on Washington . Hayes was peacefully inaugurated . Points 1 and 2 of the compromise took effect . Hayes had
already announced his support for the restoration of `` home rule , '' which would involve federal troop removal , before the election . It was not unusual ,
nor unexpected , for a president , especially one so narrowly elected , to select a cabinet member favored by the other party . Points 3 and 4 were never
enacted ; it is possible there was no firm agreement about them . Whether by informal deal or simply reassurances already in line with Hayes 's announced plans
, talks with Southern Democrats satisfied the worries of many . This prevented a Congressional filibuster that had threatened to extend resolution of the
election dispute beyond Inauguration Day 1877 . Interpretations ( edit ) C. Vann Woodward ( 1951 ) argued for an economic interpretation , saying that emerging
business and industry interests of the New South found common ground with Republican businessmen , particularly with the railroads .
</item 5>

<item 6>
Compromise of 1877
They met secretly at Wormley 's Hotel in Washington to forge a compromise with aid to internal improvements : bridges , canals and railroads wanted by the South
. But Peskin notes that no serious federal effort was made after Hayes took office to fund a railroad or provide other federal aid for improvements . An
opposing interest group representing the Southern Pacific actually thwarted Scott 's proposed Texas and Pacific scheme , and ultimately ran its own line to New
Orleans . Some historians , such as Allan Peskin , argue that the assurances offered to some Southern Democrats to prevent a filibuster were not a compromise
but a foregone conclusion , as Tilden did not command sufficient support . Peskin admits that Woodward 's interpretation had become almost universally accepted
in the nearly quarter century since he had published it . As not all terms of the agreement were met , Peskin believes there was really no deal between the
North and South in 1877 . He also suggests that Northern Democrats were more significant in quashing the filibuster than those from the South . For instance ,
Samuel J. Randall ( D - Pennsylvania ) was Speaker of the House and prevented the filibuster .
</item 6>

<item 7>
Compromise of 1877
Downs explores how Mexicanization was roundly rejected and stability was achieved . Whatever deals may or may not have taken place on the side , in formal legal
terms , the election of 1876 was not decided by such acts , but by the official vote of Congress to accept the recommendations of the Electoral Commission they
themselves had set up as a way out of the election impasse . The expectation in setting up the committee had been that its decisions would be accepted by
Congress . It was only when certain Democrats disagreed with the commission 's decisions in favor of Hayes that this arrangement was jeopardized . This
Democratic group threatened a filibuster ( opposed by Republicans and Congressional Democratic leadership as well ) that would prevent the agreed - upon vote
from taking place . Discussions of the points in the alleged compromise were related to persuading key Democrats against accepting a filibuster . The very
threat of a filibuster -- a measure used by a minority to prevent a vote -- indicates that there were already sufficient votes for accepting the commission 's
recommendations .
</item 7>

<item 8>
Compromise of 1877
U of Michigan Press . p. 211 . Jump up ^ C. Vann Woodward ( 1991 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Oxford UP .
pp. 200 -- 2 ` . Jump up ^ Downs , 2012 Jump up ^ C. Vann Woodward ( 1991 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction .
Oxford UP . pp. 200 -- 2 ` . ^ Jump up to : Allan Peskin , `` Was There a Compromise of 1877 ? '' , The Journal of American History Vol. 60 , No. 1 ( Jun. ,
1973 ) , pp. 63 - 75 , via JSTOR Jump up ^ Vincent P. DeSantis , `` Rutherford B. Hayes and the Removal of the Troops and the End of Reconstruction , '' in J.
Morgan Kousser and James M. McPherson , eds.
</item 8>

<item 9>
Compromise of 1877
( Registration required ( help ) ) . Polakoff , Keith Ian . The Politics of Inertia : The Election of 1876 and the End of Reconstruction ( 1973 ) Riddleberger ,
Patrick W. ( 1960 ) . `` The Radicals ' Abandonment of the Negro During Reconstruction '' . The Journal of Negro History . 45 ( 2 ) : 88 -- 102 . doi : 10.2307
/ 2716572 . JSTOR 2716572 -- via JSTOR . Simpson , Brooks D. `` Ulysses S. Grant and the Electoral Crisis of 1876 - 1877 , '' Hayes Historical Journal ( 1992 )
11 # 2 pp 5 -- 22 . Woodward , C. Vann ( 1951 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Oxford University Press .
External links ( edit ) Samueltilden.com : History of Samuel J. Tilden website Rbhayes.org : R.B.
</item 9>

<item 10>
Compromise of 1877
military forces from the former Confederate states . At the time , U.S. troops remained in only Louisiana , South Carolina , and Florida , but the Compromise
completed their withdrawal from the region . The appointment of at least one Southern Democrat to Hayes ' cabinet . ( David M. Key of Tennessee was appointed as
Postmaster General . ) The construction of another transcontinental railroad using the Texas and Pacific in the South ( this had been part of the `` Scott Plan
, '' proposed by Thomas A. Scott of the Pennsylvania Railroad ; he had initiated negotiations resulting in the final compromise ) . Legislation to help
industrialize the South and restore its economy following Reconstruction and the Civil War . The right to deal with blacks without northern interference . In
exchange , Democrats would accept the Republican Hayes as president by not employing the filibuster during the joint session of Congress needed to confirm the
election . Results ( edit ) After the Compromise , a few Democrats complained loudly that Tilden had been cheated . There was talk of forming armed units that
would march on Washington , but President Grant was ready for that . He tightened military security , and nobody marched on Washington . Hayes was peacefully
inaugurated .
</item 10>
=== END ITEM id=51ad445b29347956 ===

=== ITEM id=b6be23820c749c37 ===
Gold sentences and facts:
Sentence 0: `` It Ai n't Me '' is a song by Norwegian DJ Kygo and American singer Selena Gomez . It was released by Interscope Records , Sony and Ultra on 17 February 2017 as the lead single from Kygo 's first EP , Stargazing . The song was written by Kygo , Gomez , Andrew Watt , Brian Lee and Ali Tamposi . It was produced by Kygo , Watt , Ben Rice and Louis Bell . A dance - pop , electropop and tropical house song , `` It Ai n't Me '' comprises an acoustic guitar line , and a build - drop arrangement in its chorus featuring pulsing piano notes , bass , synthesizers , finger - snap claps and pan flute melodies . Gomez sings the track in a husky tone , while in the chorus her vocals are reduced to recurring syllables . The lyrics are nostalgic and narrate a past relationship ruined by alcoholism and partying too often .
  Fact 0.0: 'It Ai n't Me' is a song by Norwegian DJ Kygo and American singer Selena Gomez.
  Fact 0.1: 'It Ai n't Me' was released by Interscope Records, Sony and Ultra on 17 February 2017.
  Fact 0.2: 'It Ai n't Me' was the lead single from Kygo's first EP, Stargazing.
  Fact 0.3: 'It Ai n't Me' was written by Kygo, Selena Gomez, Andrew Watt, Brian Lee and Ali Tamposi.
  Fact 0.4: 'It Ai n't Me' was produced by Kygo, Andrew Watt, Ben Rice and Louis Bell.
  Fact 0.5: 'It Ai n't Me' is a dance-pop, electropop and tropical house song.
  Fact 0.6: 'It Ai n't Me' comprises an acoustic guitar line.
  Fact 0.7: 'It Ai n't Me' has a build-drop arrangement in its chorus.
  Fact 0.8: The chorus of 'It Ai n't Me' features pulsing piano notes, bass, synthesizers, finger-snap claps and pan flute melodies.
  Fact 0.9: Selena Gomez sings the track in a husky tone.
  Fact 0.10: In the chorus of 'It Ai n't Me', Selena Gomez's vocals are reduced to recurring syllables.
  Fact 0.11: The lyrics of 'It Ai n't Me' are nostalgic.
  Fact 0.12: The lyrics of 'It Ai n't Me' narrate a past relationship ruined by alcoholism and partying too often.

Retrieved items:
<item 1>
It Ain't Me
The song carries a calm tone and a drop common in tropical house music , but is complimented by an equal measure of electronic and pop rock elements . The track
starts with a folk - like acoustic guitar riff . It is followed immediately by Gomez singing the first verse in a husky tone . As the song develops , a steady
bass and synthesizers are heard . The production is subtle and varied with finger snaps sometimes being louder than the kick drum , and the guitar riff and
piano notes alternating in and out . In the chorus , Gomez is accompanied by background vocals that pertain to the sound of a choir . `` It Ai n't Me '' A 21 -
second sound sample from `` It Ai n't Me '' featuring the chorus to the start of the drop which Gomez sings accompanied by backing vocals , piano notes , a
guitar riff , and vocal effects . Problems playing this file ? See media help . The chorus has a new melodic arrangement and a crescendo . It is ensued by chord
- buoyed builds and drops comprising finger snaps and pan flute melodies . The same synthesizers from the verses are repeated with a piano in the background .
</item 1>

<item 2>
It Ain't Me
For the Bob Dylan song , see It Ai n't Me , Babe . `` It Ai n't Me '' Single by Kygo and Selena Gomez from the EP Stargazing Released 17 February 2017 Format
Digital download Recorded 2016 Genre Dance - pop electropop tropical house Length 3 : 40 Label Interscope Sony Ultra Songwriter ( s ) Kyrre Gørvell - Dahll
Selena Gomez Andrew Wotman Brian Lee Ali Tamposi Producer ( s ) Kygo Andrew Watt Ben Rice Louis Bell Kygo singles chronology `` Carry Me '' ( 2016 ) `` It Ai
n't Me '' ( 2017 ) `` First Time '' ( 2017 ) Selena Gomez singles chronology `` Trust Nobody '' ( 2016 ) Trust Nobody 2016 `` It Ai n't Me '' ( 2017 ) It Ai n't
Me2017 `` Bad Liar '' ( 2017 ) Bad Liar 2017 Music video `` It Ai n't Me '' on YouTube `` It Ai n't Me '' is a song by Norwegian DJ Kygo and American singer
Selena Gomez .
</item 2>

<item 3>
It Ain't Me
It Ai n't Me - wikipedia It Ai n't Me Jump to : navigation , search This article is about the Kygo and Selena Gomez song . For the Bob Dylan song , see It Ai
n't Me , Babe .
</item 3>

<item 4>
It Ain't Me
The same synthesizers from the verses are repeated with a piano in the background . Gomez 's vocals in the drop are reduced to syllables from her verses such as
a recurring `` whiskey neat '' and are accompanied by percussion instruments . `` It Ai n't Me '' is a breakup song with a theme of nostalgia . The lyrics are
about standing up for yourself in the face of a toxic relationship . Gomez narrates the regret of a previous relationship ruined by her former lover 's habits
of drinking and partying too often . The lyrics specifically address alcohol abuse , a type of alcoholism . Tamposi explained that `` It Ai n't Me '' is about a
woman finding the strength to walk away from a toxic relationship despite the pressure she receives from society to stand by her man . Although the lyrics are
sombre , the song has a distinctly uplifting and empowering sound . Release ( edit ) In early February 2017 , `` It Ai n't Me '' was registered on the American
Society of Composers , Authors and Publishers ( ASCAP ) database . Gomez first teased the single with two Instagram stories on 3 February 2017 .
</item 4>

<item 5>
It Ain't Me
It also reached the top 10 in Italy , Spain , the United Kingdom and the United States . Contents ( hide ) 1 Writing and recording 2 Music and lyrics 3 Release
4 Critical reception 5 Chart performance 6 Music video 7 Usage in media 8 Track listing 9 Credits and personnel 10 Charts 10.1 Weekly charts 10.2 Year - end
charts 11 Certifications 12 Release history 13 References 14 External links Writing and Recording ( edit ) The lyrics of `` It Ai n't Me '' were written by
Andrew Watt , Brian Lee and Ali Tamposi . With Watt previously working with Kygo , the DJ 's manager Myles Shear contacted Watt for a session with Kygo which
Watt also facilitated for Lee and Tamposi . During their session for `` It Ai n't Me '' , they initially found the track very poor , but were trying to salvage
it . Kygo and Shear suggested that they continue working on the song .
</item 5>

<item 6>
It Ain't Me
000 United States ( RIAA ) Platinum 1,000,000 sales figures based on certification alone shipments figures based on certification alone sales + streaming
figures based on certification alone Release history ( edit ) Region Date Version Format Label Ref . Italy 16 February 2017 Original Contemporary hit radio Sony
Various 17 February 2017 Single Digital download Sony Ultra United States 28 February 2017 Original Mainstream radio Ultra 7 March 2017 Rhythmic radio Sony
Ultra United Kingdom 17 March 2017 Contemporary hit radio United States 28 March 2017 Hot adult contemporary Various 7 April 2017 Tiësto 's AFTR : HRS Remix
Digital download References ( edit ) Jump up ^ https://www.pandora.com/artist/kygo-and-selena-gomez/it-aint-me-single/it-aint-me/TR9c9kdKhgc9Xcm ^ Jump up to :
Bein , Kat ( 23 February 2017 ) . `` Selena Gomez Is ' an Old Soul With a Beautiful Baby Face , ' Says ' It Ai n't Me ' Songwriter '' . Billboard . Retrieved 24
February 2016 .
</item 6>

<item 7>
It Ain't Me
Usage in Media ( edit ) The song is featured in tenth episode of the television series The Bold Type . Track listing ( edit ) Digital download `` It Ai n't Me
'' -- 3 : 40 Digital download -- Tiësto 's AFTR : HRS Remix `` It Ai n't Me '' ( Tiësto 's AFTR : HRS Remix ) -- 3 : 12 Credits and personnel ( edit ) Credits
adapted from Qobuz . Kygo -- composer , producer Selena Gomez -- composer , vocals Andrew Watt -- guitar , producer , vocal producer , composer , songwriter
Brian Lee -- composer , songwriter Ali Tamposi -- composer , songwriter Tom Coyne -- mastering engineer Serban Ghenea -- mixing engineer Ben Rice -- vocal
producer Louis Bell -- vocal producer Charts ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 )
Belgium ( Ultratop 50 Flanders ) Belgium ( Ultratop 50 Wallonia ) Canada ( Canadian Hot 100 ) Canada CHR / Top 40 ( Billboard ) Canada Hot AC ( Billboard ) 6
Colombia ( Monitor Latino Anglo ) Croatia ( HRT ) Czech Republic (
</item 7>

<item 8>
It Ain't Me
It was released by Interscope Records , Sony and Ultra on 17 February 2017 as the lead single from Kygo 's first EP , Stargazing . The song was written by Kygo
, Gomez , Andrew Watt , Brian Lee and Ali Tamposi . It was produced by Kygo , Watt , Ben Rice and Louis Bell . A dance - pop , electropop and tropical house
song , `` It Ai n't Me '' comprises an acoustic guitar line , and a build - drop arrangement in its chorus featuring pulsing piano notes , bass , synthesizers ,
finger - snap claps and pan flute melodies . Gomez sings the track in a husky tone , while in the chorus her vocals are reduced to recurring syllables . The
lyrics are nostalgic and narrate a past relationship ruined by alcoholism and partying too often . In addition to reaching number one in Croatia , Lebanon , and
Norway , the song attained a top five peak in Australia , Austria , Belgium , Canada , the Czech Republic , Denmark , Finland , France , Germany , Greece ,
Hungary , Ireland , the Netherlands , New Zealand , Poland , Portugal , Slovakia , Sweden and Switzerland .
</item 8>

<item 9>
It Ain't Me
by Brian Lee ( songwriter ) Songs about alcohol Tropical house songs Interscope Records singles Hidden categories : CS1 Norwegian - language sources ( no )
Pages containing links to subscription - only content CS1 French - language sources ( fr ) CS1 Spanish - language sources ( es ) CS1 Japanese - language sources
( ja ) CS1 Turkish - language sources ( tr ) CS1 German - language sources ( de ) CS1 Italian - language sources ( it ) CS1 Polish - language sources ( pl ) CS1
Swedish - language sources ( sv ) CS1 Portuguese - language sources ( pt ) All articles lacking reliable references Articles lacking reliable references from
January 2017 Use dmy dates from February 2017 Music infoboxes with deprecated parameters Articles with hAudio microformats Pages using div col with deprecated
parameters Singlechart usages for Australia Singlechart usages for Austria Singlechart usages for Flanders Singlechart usages for Wallonia Singlechart usages
for Billboardcanadachrtop40 Singlechart called without song Singlechart usages for Billboardcanadahotac Singlechart usages for Czech Republic Singlechart called
without artist Singlechart usages for Czechdigital Singlechart usages for Denmark Singlechart usages for Finland Singlechart usages
</item 9>

<item 10>
It Ain't Me
Enter It Ai n't Me in the search field and then press Enter . Jump up ^ `` American single certifications -- Kygo & Selena Gomez -- It Ai n't Me '' . Recording
Industry Association of America . Retrieved 7 June 2017 . If necessary , click Advanced , then click Format , then select Single , then click SEARCH Jump up ^
`` Kygo & Selena Gomez - It Ai n't Me RadioDate '' . Radioairplay.fm. 16 February 2017 . Retrieved 9 September 2017 . Jump up ^ ( 1 ) Jump up ^ `` BBC - Radio 1
- Playlist '' . BBC Radio 1 . 17 March 2017 . Archived from the original on 18 March 2017 . Retrieved 21 June 2017 . Jump up ^ `` Music Meeting for March 28 ''
. Play IGA . Archived from the original on 28 March 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Kygo Discography Studio albums Cloud Nine
( 2016 ) Kids in Love ( 2017 ) EPs Stargazing ( 2017
</item 10>
=== END ITEM id=b6be23820c749c37 ===

=== ITEM id=a67248097068b785 ===
Gold sentences and facts:
Sentence 0: John James McCoy is a fictional character in the television drama Law & Order . He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order , four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the made - for - TV movie Exiled .
  Fact 0.0: John James McCoy is a fictional character in the television drama Law & Order.
  Fact 0.1: John James McCoy was created by Michael S. Chernuchin.
  Fact 0.2: John James McCoy was portrayed by Sam Waterston from 1994 until the end of the series in 2010.
  Fact 0.3: John James McCoy is the second-longest tenured character on the show, with 16 seasons.
  Fact 0.4: John James McCoy's tenure ranks just after Lt. Anita Van Buren, who had 17 seasons.
  Fact 0.5: Lt. Anita Van Buren was portrayed by S. Epatha Merkerson.
  Fact 0.6: John James McCoy appeared in 368 episodes of Law & Order.
  Fact 0.7: John James McCoy appeared in four episodes of Law & Order: Special Victims Unit.
  Fact 0.8: John James McCoy appeared in two episodes of Law & Order: Trial by Jury.
  Fact 0.9: John James McCoy appeared in two episodes of Homicide: Life on the Street.
  Fact 0.10: John James McCoy appeared in the made-for-TV movie Exiled.
Sentence 1: Jack McCoy brings 24 years of experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends -- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick , and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics occasionally incur negative publicity for the DA 's office . His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial .
  Fact 1.0: Jack McCoy brings 24 years of experience with him.
  Fact 1.1: Jack McCoy is appointed executive assistant district attorney by Adam Schiff.
  Fact 1.2: Adam Schiff is played by Steven Hill.
  Fact 1.3: Jack McCoy is appointed in the season-five premiere episode 'Second Opinion'.
  Fact 1.4: Jack McCoy quickly establishes himself as a more unconventional, ruthless litigator than his predecessor.
  Fact 1.5: Jack McCoy's predecessor is Ben Stone.
  Fact 1.6: Ben Stone is played by Michael Moriarty.
  Fact 1.7: Jack McCoy often bends trial rules to get convictions.
  Fact 1.8: Jack McCoy sometimes breaks trial rules to get convictions.
  Fact 1.9: Jack McCoy finds tenuous rationales for charging defendants with crimes when the original charges fail to stick.
  Fact 1.10: Jack McCoy charges innocent people to frighten them into testifying against others.
  Fact 1.11: Jack McCoy is found in contempt of court 80 times for such behavior.
  Fact 1.12: Jack McCoy's tactics occasionally incur negative publicity for the DA's office.
  Fact 1.13: Jack McCoy's underlying motivation is not corruption, he maintains.
  Fact 1.14: Jack McCoy's underlying motivation is a sincere desire to see justice done.
  Fact 1.15: Jack McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions.
  Fact 1.16: Jack McCoy pursues those defendants with just as much determination as his more mundane cases.
  Fact 1.17: Jack McCoy's aggressive actions in the courts have earned him the nickname 'Hang 'em High McCoy'.
  Fact 1.18: Jack McCoy has developed a reputation with both colleagues and rival attorneys.
  Fact 1.19: Jack McCoy was once referred to as 'the top of the legal food chain' by a rival attorney during a trial.
Sentence 2: `` Entitled '' ( onscreen ) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) .
  Fact 2.0: The episode 'Entitled' (onscreen) is where McCoy makes his first appearance on the series.
  Fact 2.1: McCoy works with his assistant, Abbie Carmichael.
  Fact 2.2: Abbie Carmichael is played by Angie Harmon.
  Fact 2.3: McCoy assists the Special Victims Unit in solving a cold case.
  Fact 2.4: Briscoe handled the cold case years ago.
  Fact 2.5: Briscoe handled the cold case with his former partner, Detective Mike Logan.
  Fact 2.6: Detective Mike Logan is played by Chris Noth.

Retrieved items:
<item 1>
Jack McCoy
He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured
character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order
, four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the
made - for - TV movie Exiled . Waterston 's performance as McCoy on the New York - based series was so popular that it resulted in him being declared a ``
Living Landmark '' by the New York Landmarks Conservancy , along with fellow longtime series cast member Jerry Orbach ( who portrayed the popular police
detective Lennie Briscoe for 12 years ) . Contents ( hide ) 1 Character overview 2 Personal life 3 Notable conflicts 4 In Law & Order : Special Victims Unit 4.1
As executive assistant district attorney 4.1.
</item 1>

<item 2>
Jack McCoy
His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after
defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive
actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival
attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial . Following the 17th season ( 2006 -- 2007 ) ,
Jack McCoy was appointed interim district attorney , taking over from Arthur Branch ( Fred Thompson ) . McCoy 's appearance on Law & Order : Special Victims
Unit on the November 13 , 2007 , episode `` Blinded '' , marked his first appearance in the Law & Order universe as district attorney . The replacement for his
former position is Michael Cutter ( Linus Roache ) , a prosecutor with a penchant for recklessness not unlike McCoy 's own in his younger days . This
occasionally presents political difficulties for the new district attorney .
</item 2>

<item 3>
Jack McCoy
Jack McCoy - wikipedia Jack McCoy Jump to : navigation , search For the politician and labor activist , see Jack E. McCoy . Jack McCoy Law & Order character
First appearance `` Second Opinion '' `` Entitled '' ( SVU ) Last appearance `` Rubber Room '' ( Law & Order ) `` The Undiscovered Country '' ( SVU ) Portrayed
by Sam Waterston Time on show 1994 -- 2010 , 2018 Seasons 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 Credited appearances 368
episodes ( L&O ) 2 episodes ( HLOTS ) 4 episodes ( SVU ) 2 episodes ( TBJ ) 376 episodes ( total ) Preceded by Ben Stone ( EADA ) Arthur Branch ( DA ) ( L&O )
Succeeded by Michael Cutter ( EADA ) Information Family Rebecca McCoy ( daughter ) John James McCoy is a fictional character in the television drama Law & Order
.
</item 3>

<item 4>
Jack McCoy
Reception ( edit ) Entertainment Weekly television critic Ken Tucker has praised Law & Order 's creator Dick Wolf for putting McCoy at the center of `` some of
the best episodes of the immortal series ' 19th season . '' Tucker elaborates how the character , riding `` herd over a couple of stubborn young bucks --
assistant DAs Mike Cutter ( Linus Roache ) and Connie Rubirosa ( Alana de la Garza ) -- McCoy argues , bellows orders , and croaks with outrage when his charges
disobey his legal advice . ''
</item 4>

<item 5>
Jack McCoy
wikipedia.org/w/index.php?title=Jack_McCoy&oldid=838021970 '' Categories : Law & Order characters Fictional lawyers Fictional American people of Irish descent
Fictional characters introduced in 1994 Fictional district attorneys Fictional victims of child abuse Crossover characters in television Hidden categories :
Pages using deprecated image syntax Articles using Infobox character with multiple unlabeled fields Talk Contents About Wikipedia Deutsch Hrvatski 日本 語
Português Srpskohrvatski / српскохрватски Edit links This page was last edited on 24 April 2018 , at 13 : 14 . About Wikipedia
</item 5>

<item 6>
Jack McCoy
NBC . Jump up ^ `` Access Nation '' . Law & Order . Season 12 . Episode 15 . February 27 , 2002 . NBC . Jump up ^ `` Rebels '' . Law & Order . Season 6 .
Episode 2 . September 27 , 1995 . NBC . Jump up ^ `` Embedded '' . Law & Order . Season 14 . Episode 8 . November 19 , 2003 . NBC . Jump up ^ `` Burned '' . Law
& Order . Season 8 . Episode 9 . December 10 , 1997 . NBC . Jump up ^ `` Deadbeat '' . Law & Order . Season 7 . Episode 7 . November 13 , 1996 . NBC . Jump up ^
`` Good Faith '' . Law & Order . Season 17 . Episode 17 . March 30 , 2007 . NBC . Jump up ^ `` Angel '' . Law & Order . Season 6 . Episode 8 . November 29 ,
1995 . NBC .
</item 6>

<item 7>
Jack McCoy
1 Season one 4.1. 2 Season two 4.2 As district attorney 4.2. 1 Season 9 4.2. 2 Season 10 4.2. 3 Season 11 4.2. 4 Season 12 4.2. 5 Season 13 4.2. 6 Season 19 5
Reception 6 District attorney 's office timeline 7 Appearances on other TV shows 8 References Character overview ( edit ) Jack McCoy brings 24 years of
experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second
Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends
-- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick
, and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics
occasionally incur negative publicity for the DA 's office .
</item 7>

<item 8>
Jack McCoy
By the end of the 18th season , however , McCoy discovers that Shalvoy is involved in a prostitution scandal that is tied to a murder case he is prosecuting .
Angered and disappointed , McCoy orders Cutter to start investigating Shalvoy , who retaliates by lending his support to McCoy 's opponent in the election . In
the last episode of season 19 , `` The Drowned and the Saved '' , Cutter uncovers evidence that Shalvoy tried to buy a Senate seat for his wife , and threatens
him with public disgrace unless he resigns . McCoy 's opponent suddenly has no patron , giving McCoy good prospects for victory . In the opening episode of
season 20 , McCoy was revealed to have won the election ; he serves as DA for the remainder of this final season of Law & Order . Personal Life ( edit ) While
he is a brilliant legal mind , McCoy has more than a few personal demons . He was abused by his father , an Irish Chicago policeman who had also beat Jack 's
mother , and who eventually died of cancer . McCoy says that his determination and unyielding work ethic are a byproduct of having been harshly punished by his
father for losing at anything .
</item 8>

<item 9>
Jack McCoy
Although McCoy insists that he is a prosecutor , and not a politician , McCoy replaces Branch as the DA immediately thereafter . In Law & Order : Special
Victims Unit ( edit ) McCoy has appeared in four episodes of Law & Order : Special Victims Unit ; however , he is often referred to in the series , and his
actions affect the ADAs working with the Special Victims Unit . As executive assistant district attorney ( edit ) Season one ( edit ) `` Entitled '' ( onscreen
) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit
in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) . Season two ( edit ) `` Baby Killer '' (
mention ) : After a murder intertwines with ADA Alexandra Cabot 's ( Stephanie March ) sex - crimes case , Chief Assistant District Attorney Charlie Phillips (
Jeffrey DeMunn ) , Lewin 's second in command , hands the murder case to McCoy . As district attorney ( edit ) Season 9 ( edit ) `` Blinded '' ( onscreen ) :
McCoy had been promoted to district attorney .
</item 9>

<item 10>
Jack McCoy
District attorney 's office timeline ( edit ) Time period Executive assistant district attorney ( EADA ) Assistant district attorney ( ADA ) District attorney (
DA ) 1994 -- 1996 Jack McCoy Claire Kincaid Adam Schiff 1996 -- 1998 Jamie Ross 1998 -- 2000 Abbie Carmichael 2000 -- 2001 Nora Lewin 2001 -- 2002 Serena
Southerlyn 2002 -- 2005 Arthur Branch 2005 -- 2006 Alexandra Borgia 2006 -- 2007 Connie Rubirosa 2008 -- 2010 Michael Cutter Jack McCoy Appearances on other TV
shows ( edit ) Homicide : Life on the Street Season 6 Episode 5 : `` Baby ,
</item 10>
=== END ITEM id=a67248097068b785 ===

=== ITEM id=e13d8d7d58167ccd ===
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
Crazy Eyes ( character ) - wikipedia Crazy Eyes ( character ) Jump to : navigation , search Suzanne `` Crazy Eyes '' Warren Orange Is the New Black character
Uzo Aduba as Suzanne `` Crazy Eyes '' Warren in a promotional poster for season 2 of Orange Is the New Black First appearance `` I Was n't Ready '' Season 1 ,
episode 1 July 11 , 2013 Portrayed by Uzo Aduba Eden Wiggins ( at age 10 ) Taliyah Whitaker ( at age 5 ) Information Significant other ( s ) Maureen Kukudio (
ex-girlfriend ) Parents Pat and Dennis Warren Suzanne `` Crazy Eyes '' Warren is a fictional character played by Uzo Aduba on the Netflix series Orange Is the
New Black . Warren is portrayed as intelligent , but lacking in social skills , and prone to spiral into emotional outbursts when agitated . The character is
the only role that has received Emmy Award recognition both in the comedy and drama genres from the same show and only the second character to earn Emmy
recognition in both genres .
</item 1>

<item 2>
Crazy Eyes (character)
September 17 , 2017 . Retrieved September 20 , 2017 . ( hide ) Orange Is the New Black Characters Piper Chapman Alex Vause Crazy Eyes Nicky Nichols Tiffany
Doggett Lorna Morello Miss Rosa Poussey Washington Seasons `` Lesbian Request Denied '' 5 Related Piper Kerman Orange Is the New Black : My Year in a Women 's
Prison `` You 've Got Time '' Retrieved from `` https://en.wikipedia.org/w/index.php?title=Crazy_Eyes_(character)&oldid=814872296 '' Categories : Orange Is the
New Black characters Fictional adoptees Fictional African - American people Fictional characters introduced in 2013 Fictional criminals Fictional lesbians
Fictional prison escapees Fictional writers Fictional LGBT characters in television Fictional inmates in American federal prisons Hidden categories : Use mdy
dates from September 2016 Articles needing additional references from June 2017 All articles needing additional references Talk Contents About Wikipedia Türkçe
Edit links This page was last edited on 11 December 2017 , at 12 : 19 . About Wikipedia
</item 2>

<item 3>
Crazy Eyes (character)
Jump up ^ Schaefer , Megan ( June 3 , 2014 ) . `` ' Orange Is The New Black ' Season 1 Recap : Everything To Know Before The Season 2 Premiere '' .
International Business Times . Retrieved September 6 , 2014 . Jump up ^ Foster , Lewis ( June 19 , 2014 ) . `` Visual Summary : Orange Is The New Black Season 1
'' . Paste . Retrieved September 6 , 2014 . Jump up ^ O'Sullivan , Erin ( August 14 , 2013 ) . `` Orange Is The New Black : Uzo Aduba ( Suzanne Warren )
Promoted To Series Regular '' . Access Hollywood . Retrieved September 6 , 2014 . Jump up ^ Henderson , Danielle ( August 16 , 2013 ) . `` Orange Is the New
Black Episode Nine Recap : We Finally Experience the SHU '' . Vulture . Retrieved September 6 , 2014 . Jump up ^ `` Orange is the New Black Season 1 Episode 9 :
F * cksgiving : Episode Cast & Crew '' . TV.com .
</item 3>

<item 4>
Graph query row:
entity: Suzanne Warren
relationship: RELATED_TO
detail: character on
related: Orange Is the New Black
related_detail: TV show
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
It is mentioned that she took a plea deal . When the inmates take advantage of a temporary lack of security and enjoy the nearby lake and woods , Kukudio
insists that she and Suzanne run away together . Suzanne refuses and their relationship becomes rocky . In `` People Persons , CO Humphrey commands the two of
them to fight . Suzanne rejects this but Maureen , who wants to fight , provokes her by launching into a stream of insults . Suzanne fully retaliates by
throwing Maureen to the ground and beating her repeatedly , until the Dominicans pull her off . This fight leads her to extreme guilt and a wish to punish
herself . She demonstrates this especially when a peaceful protest in the jail cafeteria turns into a chaotic scene and Poussey Washington dies by suffocation .
Whilst the inmates grieve over her death , Suzanne believes that Poussey is alive and tries numerous times to suffocate herself . Finally , she is sent to
Medical and is placed in a bed next to Kukudio . Season 5 ( edit ) In season five , Suzanne continues to recover from her injuries and also goes out of her way
to help Maureen , whose injuries are severe .
</item 6>

<item 7>
Crazy Eyes (character)
Eventually , she turns to new prison counsellor Berdie Rogers 's drama class . There , she begins writing a surrealistic science - fiction erotica series called
`` The Time Hump Chronicles '' which , although considered obscene and strange by both Rogers and Taystee , becomes an instant hit amongst the inmates .
Although they prove to be therapeutic for her mental health , Suzanne becomes stressed and irritated with the constant harassment by the other inmates for more
material , the persistent ideas that are being floated past her and the fan - fiction that is left at her bunk or lying around . Soon , some of the extracts
find their way into the hands of the COs , resulting in Officer Donaldson being mocked by both his colleagues and the inmates alike upon the realization that he
is the inspiration for one of the characters . This leads to Rogers ' suspension . Meanwhile , Suzanne is taken by surprise when she discovers that one of her
more prolific fans , Maureen Kukudio , is actually interested in her romantically . Suzanne seeks advice from Lorna about how to respond ; she admits she finds
Maureen attractive , but she has never had a girlfriend before and subsequently feels she must back away from a sexual liaison .
</item 7>

<item 8>
Crazy Eyes (character)
In the season finale , Suzanne assists Poussey and Taystee in caring for Brook Soso , whom they discover unconscious from a drug overdose , and develops a
closer bond with Maureen in the final scene , owing to a misunderstanding over a turtle that bites Maureen 's foot . Season 4 ( edit ) In season 4 it is
revealed that Suzanne 's crime was kidnapping and accidental manslaughter of a child she grew friendly with while working as a store greeter . Believing that
she and the boy were simply being friends , she did not realize the implications of her actions when she came across him playing in the park and invited him to
her apartment for ice cream and video games . The boy went willingly , but soon became frightened and wanted to call his dad . While Suzanne is out of the room
preparing for what she plans as more fun , he calls 911 . Suzanne becomes agitated as she tells him that 911 should only be called when there is an emergency .
The child panics and tries to get away . He goes onto the fire escape and falls to his death . It is mentioned that she took a plea deal .
</item 8>

<item 9>
Crazy Eyes (character)
Suzanne gives a statement to prison officials confessing to the attack , but inmate counselor Sam Healy ( Michael J. Harney ) produces evidence that exonerates
her . When she realizes that Vee used her , she breaks into tears . In episode 3 , `` Hugs Can Be Deceiving '' , Suzanne 's backstory is revealed via a
flashback . She was adopted by white parents and raised in the suburbs , but her mental instability alienated her classmates and their parents alike . Her
adoptive mother pushed her to excel at everything she did to prove that she was as good as everyone else ; ironically , the constant pressure only made Suzanne
's psychological problems worse . In the same episode , it is revealed that Suzanne had run outside and knocked Chapman unconscious after Chapman had beaten
Tiffany `` Pennsatucky '' Doggett ( Taryn Manning ) unconscious at the end of season 1 . Season 3 ( edit ) In season 3 , Taystee has her hands full trying to
control Suzanne , who is still grieving Vee 's death and is unstable , prone to losing her temper and constantly fighting with Poussey .
</item 9>

<item 10>
Crazy Eyes (character)
Deadline.com . Retrieved September 5 , 2014 . Jump up ^ Kohan , Jenji ( writer / creator ) ; Herrmann , Tara ( producer ) ; Burley , Mark ( producer ) ( May 13
, 2014 ) . Orange Is the New Black : Season 1 -- `` I Was n't Ready '' audio commentary ( Blu - ray ) . Lionsgate . Event occurs at 37 : 00 -- 37 : 30 . Jump up
^ `` Uzo Aduba '' . Internet Movie Database . Retrieved June 12 , 2015 . Jump up ^ `` Orange is the New Black Season 2 Episode 3 : Hugs Can Be Deceiving '' .
TV.com . Retrieved September 5 , 2014 . Jump up ^ Potts , Kimberly ( June 18 , 2014 ) . `` The 10 Best ' Orange Is the New Black ' Character Backstories '' .
Yahoo ! TV . Retrieved September 5 , 2014 . Jump up ^ White , Chelsea F.C. ( October 2 , 2013 ) . `` Spot the difference !
</item 10>
=== END ITEM id=e13d8d7d58167ccd ===

=== ITEM id=6fbd8854814baf57 ===
Gold sentences and facts:
Sentence 0: A defensive three - second violation , also known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a guarding position . A three - second count is suspended if :
  Fact 0.0: A defensive three-second violation is also known as illegal defense.
  Fact 0.1: A defensive three-second violation is a basketball rules infraction in the National Basketball Association (NBA).
  Fact 0.2: The defensive three-second rule was introduced in the 2001-2002 season.
  Fact 0.3: A defensive three-second violation is assessed when a member of the defending team spends more than three seconds in the free throw lane while not actively guarding an opponent.
  Fact 0.4: The free throw lane is also known as the 16-foot lane.
  Fact 0.5: The free throw lane is colloquially known as 'in the paint'.
  Fact 0.6: To be considered actively guarding, a defender must be within arm's length of an opponent.
  Fact 0.7: To be considered actively guarding, a defender must be in a guarding position.
  Fact 0.8: A three-second count is suspended under conditions listed after the word 'if'.

Retrieved items:
<item 1>
Defensive three-second violation
corners Motion Princeton Run and gun Grinnell System Nellie Ball Small ball Triangle Defense 1 - 2 - 1 - 1 1 -- 3 -- 1 2 -- 3 zone Amoeba Box - and - one Double
team Full - court press Hack - a-Shaq Jordan Rules Line Man - to - man Match - up zone Triangle - and - two Zone Equipment Backboard Ball Breakaway rim Court
Half court Key Net Possession arrow Whistle Clothing Basketball sleeve Finger sleeve Air Jordan Chuck Taylor All - Stars Miscellaneous 100 - point scorers 50 --
40 -- 90 club AAU All - Americans AAU champions Mikan Drill Player tracking Shootaround Tip drill Winning streaks This basketball - related article is a stub .
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Defensive_three-second_violation&oldid=803130800 ''
Categories : Basketball penalties Basketball stubs Hidden categories : Wikipedia articles needing clarification from November 2015 All stub articles Talk About
Wikipedia Français Edit links This page was last edited on 30 September 2017 , at 17 : 04 . About Wikipedia
</item 1>

<item 2>
2017–18 NBA season
The regular season started eight days earlier than the previous season . This move by the NBA aims for fewer `` back - to - back '' games , even fewer than the
previous season which had already reached an all - time low to help address teams that are `` resting '' their superstars . The NBA decreased the maximum number
of timeouts in a regulation game from 18 to 14 . The three team timeouts during the last two minutes of the game were also decreased to two per team in the last
three minutes to improve the flow and pace of the game . The trade deadline was moved from four days after the NBA All - Star Game to ten days before it , so
teams can settle their rosters before the All - Star break . Free - throw shooters will be assessed a delay - of - game violation if they venture outside of the
three - point line in between attempts . The NBA also added two - way contracted players , signifying an increasing partnership between the NBA and the newly
branded NBA G League . Referees can now assess dangerous closeouts of defenders to jump shooters with little space to land for a technical or a flagrant foul .
</item 2>

<item 3>
Defensive three-second violation
The WNBA started using it in 2013 . References ( edit ) Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , B. Retrieved July 25 , 2010 Jump up
^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , D. Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , E.
Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , Penalty . Retrieved July 25 , 2010 Jump up ^ Blinebury , Fran (
June 8 , 2006 ) . `` Rule changes have NBA back in the fast lane '' . Houston Chronicle . Retrieved July 26 , 2010 . Jump up ^
http://www.sportingcharts.com/dictionary/nba/defensive-three-second-violation.aspx Jump up ^ Cruz , Agnes ( January 24 , 2003 ) . `` PBA get fresh look this
season '' . Arab News .
</item 3>

<item 4>
NBC Sunday Night Football
Then the banner returns to normal and show the change in the team 's score . Additionally ( beginning with Week 9 ) , timeout indicators were added below each
team 's respective scores . For the 2010 season , the timeout indicators were changed to three white trapezoids below the team abbreviations , and the play
clock was moved from above the team in possession of the football to above the game clock ( for the final two minutes of regulation and if necessary , overtime
) . The down markers also changed in 2010 , which is now featuring the team logo next to the down marker . 2012 -- 14 ( edit ) The scoring banner used from 2012
to 2014 , starting with the 2011 NFL Wild Card playoffs to Week 17 of the 2014 NFL regular season . Notice the addition of timeout indicators at the bottom ,
and the cleaner and larger looking font in the team 's initials .
</item 4>

<item 5>
American League
Most umpires made the switch to the inside protector , led by Don Denkinger in 1975 and Jim Evans the next year , although several did not , including Bill
Haller , Lou DiMuro , George Maloney , and Jerry Neudecker , who became the last MLB umpire to use the outside protector in 1985 . In 1994 , the league , along
with the National League , reorganized again , into three divisions ( East , West , and Central ) and added a third round to the playoffs in the form of the
American League Division Series , with the best second - place team advancing to the playoffs as a wild - card team , in addition to the three divisional
champions . In 1998 , the newly franchised Tampa Bay Devil Rays joined the league , and the Arizona Diamondbacks joined the National League : i.e. , each league
each added a fifteenth team . An odd number of teams per league meant that at least one team in each league would have to be idle on any given day , or
alternatively that odd team out would have had to play an interleague game against its counterpart in the other league .
</item 5>

<item 6>
2017–18 NBA season
Informally dubbed as the `` Zaza Pachulia rule '' . Also , officials can now assess if a player is in a shooting motion when determining if a perimeter foul is
a shooting or a common foul . The type of foul will be determined on the sequencing of the player 's movement if the foul happens before or after the player
starts a shooting motion . Additionally , they can also assess an offensive foul if an offensive player intentionally lunges into the defender or locks arms to
create contact , a rule informally dubbed as the `` Harden rule '' . On September 28 , 2017 , the NBA board of governors approved the changes to the draft
lottery system to discourage teams from losing on purpose . Starting with the 2018 -- 19 NBA season , the teams with the three worst records will have equal
odds at the # 1 pick . In addition , the revised draft lottery will consist of selecting the top four teams instead of the usual three that occurred for over
two decades . The approval for the change was nearly unanimous , with only the Oklahoma City Thunder voting against the change and the Dallas Mavericks
abstaining from voting .
</item 6>

<item 7>
Defensive three-second violation
`` PBA get fresh look this season '' . Arab News . Retrieved January 13 , 2014 . Jump up ^ `` WNBA Announces Rules Changes for 2013 Season '' . WNBA . Retrieved
Dec 16 , 2012 . Basketball ( outline ) General topics History of basketball James Naismith Variations of basketball Leagues Statistics Glossary of terms Index
of articles Rules Bonus Jump ball Officials Turnover Violation 3 seconds Offense Defense 5 seconds Basket interference Carrying Double dribble Goaltending Shot
clock Traveling Foul Flagrant Personal Technical Game play Air ball Alley - oop Assist Backboard shattering Ball hog Block Buzzer beater Cherry picking Dribble
Crossover Dunk Euro step Fadeaway Fast break Fly Flop Jump shot Layup Finger roll Field goal Four - point play Free throw Hook shot Moves Pick and roll
Positions Posterized Playbook Rebound Point Screen Back screen Slashing Steal Three - pointer Three - point play Uncontested shot Strategy General Sixth man
Offense Continuity offense Flex Shuffle UCLA High Post Wheel Dribble drive motion Four corners Motion Princeton Run and gun Grinnell System Nellie Ball Small
ball Triangle Defense 1
</item 7>

<item 8>
Defensive three-second violation
The offense receives one free throw and retains possession of the ball . Prior to the 2001 -- 2002 NBA season , any form of zone defense was considered an
illegal defense violation and resulted in a warning on the first violation and then a technical foul for any subsequent violations . However , the defensive
three - second violation makes it difficult for NBA defenses to play zone , since zone defenses usually position a player in the middle of the key to stop
penetration . The Philippine Basketball Association used to follow the illegal defense rule until the 2003 PBA season when it was abolished . The high school
game does not use this rule , nor does European basketball . The WNBA started using it in 2013 .
</item 8>

<item 9>
2017–18 NBA season
`` D - League salaries , two - way contracts increase NBA Draft early entries '' . CollegeBasketballTalk . Retrieved September 20 , 2017 . Jump up ^ Mahoney ,
Brian . `` NBA referees can rule closeouts as flagrants or technical fouls '' . NBA.com . Retrieved September 22 , 2017 . Jump up ^ Bushnell , Henry . `` The
NBA is cracking down on dangerous Zaza-esque closeouts '' . sports.yahoo.com . Retrieved September 22 , 2017 . Jump up ^ Boone , Kyle ( September 22 , 2017 ) .
`` The NBA is finally cracking down on James Harden 's foul - drawing antics '' . CBSSports.com . Retrieved October 4 , 2017 . Jump up ^ `` NBA implementing '
Zaza Pachulia , ' ' James Harden ' rules '' . NBCSports.com . September 21 , 2017 . Retrieved October 4 , 2017 . Jump up ^ `` NBA Board Of Governors Approves
Changes To Draft Lottery System '' . NBA.com .
</item 9>

<item 10>
Defensive three-second violation
Defensive three - second violation - Wikipedia Defensive three - second violation Jump to : navigation , search A defensive three - second violation , also
known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is
assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in
the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a
guarding position . A three - second count is suspended if : a player is in the act of shooting there is a loss of team control the defender is actively
guarding an opponent the defender clears the painted area it is imminent the defender will become legal In addition , a player guarding an opponent with the
ball may be in the paint without actively guarding the opponent . The team committing a defensive three - second violation is assessed a team technical foul .
The offense receives one free throw and retains possession of the ball .
</item 10>
=== END ITEM id=6fbd8854814baf57 ===

=== ITEM id=7f9f82ce5208e49a ===
Gold sentences and facts:
Sentence 0: The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle Eastern politics . Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League from 1979 until 1989 .
  Fact 0.0: Time has elapsed since the Camp David Accords.
  Fact 0.1: The passage of time has left no doubt about the enormous ramifications of the Camp David Accords on Middle Eastern politics.
  Fact 0.2: The perception of Egypt within the Arab world changed.
  Fact 0.3: The change in perception of Egypt was the most notable effect.
  Fact 0.4: Egypt had the most powerful of the Arab militaries.
  Fact 0.5: Egypt had a history of leadership in the Arab world under Nasser.
  Fact 0.6: Egypt had more leverage than any other Arab state to advance Arab interests.
  Fact 0.7: Egypt was subsequently suspended from the Arab League.
  Fact 0.8: Egypt's suspension from the Arab League lasted from 1979 until 1989.

Retrieved items:
<item 1>
Camp David Accords
Accompanied by their capable negotiating teams and with their respective interests in mind , both leaders converged on Camp David for 13 days of tense and
dramatic negotiations from 5 to 17 September 1978 . By all accounts , Carter 's relentless drive to achieve peace and his reluctance to allow the two men to
leave without reaching an agreement are what played the decisive role in the success of the talks . Carter 's advisers insisted on the establishment of an
Egyptian - Israeli agreement which would lead to an eventual solution to the Palestine issue . They believed in a short , loose , and overt linkage between the
two countries amplified by the establishment of a coherent basis for a settlement . However , Carter felt they were not `` aiming high enough '' and was
interested in the establishment of a written `` land for peace '' agreement with Israel returning the Sinai Peninsula and West Bank . Numerous times both the
Egyptian and Israeli leaders wanted to scrap negotiations , only to be lured back into the process by personal appeals from Carter .
</item 1>

<item 2>
Camp David Accords
The efforts initially focused on a comprehensive resolution of disputes between Israel and the Arab countries , gradually evolving into a search for a bilateral
agreement between Israel and Egypt . Upon assuming office on January 20 , 1977 , President Carter moved to rejuvenate the Middle East peace process that had
stalled throughout the 1976 presidential campaign in the United States . Following the advice of a Brookings Institution report , Carter opted to replace the
incremental , bilateral peace talks which had characterized Henry Kissinger 's shuttle diplomacy following the 1973 Yom Kippur War with a comprehensive ,
multilateral approach . The Yom Kippur War further complicated efforts to achieve the objectives written in United Nations Security Council Resolution 242 .
Israel 's Prime Minister Yitzhak Rabin and his successor , Menachem Begin , were both skeptical of an international conference . While Begin , who took office
in May 1977 , officially favored the reconvening of the conference , perhaps even more vocally than Rabin , and even accepted the Palestinian presence , in
actuality the Israelis and the Egyptians were secretly formulating a framework for bilateral talks .
</item 2>

<item 3>
Camp David Accords
Camp David Accords - wikipedia Camp David Accords Jump to : navigation , search Camp David Accords Framework for Peace in the Middle East and Framework for the
Conclusion of a Peace Treaty between Egypt and Israel Celebrating the signing of the Camp David Accords : Menachem Begin , Jimmy Carter , Anwar Sadat Type
Bilateral treaty Signed 17 September 1978 ( 1978 - 09 - 17 ) Location Washington , DC , United States Original signatories Egypt Israel Signatories Menachem
Begin ( Prime Minister of Israel ) Anwar Sadat ( President of Egypt ) Jimmy Carter ( President of the United States ) Ratifiers Egypt Israel Language English
French The Camp David Accords were signed by Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin on 17 September 1978 , following twelve
days of secret negotiations at Camp David . The two framework agreements were signed at the White House , and were witnessed by United States President Jimmy
Carter . The second of these frameworks ( A Framework for the Conclusion of a Peace Treaty between Egypt and Israel ) led directly to the 1979 Egypt -- Israel
Peace Treaty .
</item 3>

<item 4>
Camp David Accords
The issues of Israeli settlement withdrawal from the Sinai and the status of the West Bank created what seemed to be an impasse . In response , Carter had the
choice of trying to salvage the agreement by conceding the issue of the West Bank to Begin , while advocating Sadat 's less controversial position on the
removal of all settlements from the Sinai Peninsula . Or he could have refused to continue the talks , reported the reasons for their failure , and allowed
Begin to bear the brunt of the blame . Carter chose to continue and for three more days negotiated . During this time , Carter even took the two leaders to the
nearby Gettysburg National Military Park in the hopes of using the American Civil War as a simile to their own struggle . Consequently , the 13 days marking the
Camp David Accords were considered a success . Partly due to Carter 's determination in obtaining an Israeli - Egyptian agreement , a full two - week pledge to
a singular international problem . Additionally , Carter was beneficiary to a fully pledged American foreign team . Likewise , the Israeli delegation had a
stable of excellent talent in Ministers Dayan and Weizman and legal experts Dr.
</item 4>

<item 5>
Camp David Accords
Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab
world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League
from 1979 until 1989 . When the Camp David accords were signed , Jordan 's King Hussein saw it as a slap to the face when Sadat volunteered Jordan 's
participation in deciding how functional autonomy would work . More specifically , Sadat effectively said that Jordan would have a role in how the West Bank
would be administered . Like the Rabat Summit Resolution , the Camp David Accords circumscribed Jordan 's objective to reassert its control over the West Bank .
Focusing as it did on Egypt , the Carter administration accepted Sadat 's claim that he could deliver Hussein . However , with Arab world opposition building
against Sadat , Jordan could not risk accepting the Accords without the support from powerful Arab neighbours , like Iraq , Saudi Arabia , and Syria . Hussein
consequently felt diplomatically snubbed .
</item 5>

<item 6>
Camp David Accords
According to The Continuum Political Encyclopedia of the Middle East : The normalization of relations ( between Israel and Egypt ) went into effect in January
1980 . Ambassadors were exchanged in February . The boycott laws were repealed by Egypt 's National Assembly the same month , and some trade began to develop ,
albeit less than Israel had hoped for . In March 1980 regular airline flights were inaugurated . Egypt also began supplying Israel with crude oil `` . According
to Kenneth Stein in Heroic Diplomacy : Sadat , Kissinger , Carter , Begin , and the Quest for Arab -- Israeli Peace : The Accords were another interim agreement
or step , but negotiations that flowed from the Accords slowed for several reasons . These included an inability to bring the Jordanians into the discussions ;
the controversy over settlements ; the inconclusive nature of the subsequent autonomy talks ; domestic opposition sustained by both Begin and Sadat and , in
Sadat 's case , ostracism and anger from the Arab world ; the emergence of a what became a cold peace between Egypt and Israel ; and changes in foreign policy
priorities including discontinuity in personnel committed to sustaining the negotiating process ( . )
</item 6>

<item 7>
Camp David Accords
Lastly , the biggest consequence of all may be in the psychology of the participants of the Arab - Israeli conflict . The success of Begin , Sadat , and Carter
at Camp David demonstrated to other Arab states and entities that negotiations with Israel were possible -- that progress results only from sustained efforts at
communication and cooperation . Despite the disappointing conclusion of the 1993 Oslo Accords between the PLO and Israel , and even though the 1994 Israel --
Jordan peace treaty has not fully normalized relations with Israel , both of these significant developments had little chance of occurring without the precedent
set by Camp David . Public support Although most Israelis supported the Accords , the Israeli settler movement opposed them because Sadat would not agree to a
treaty in which Israel had any presence in the Sinai Peninsula at all , Israel had to withdraw from the entire Sinai Peninsula . Israeli settlers tried to
prevent the government from dismantling their settlements . In Israel , there is lasting support of the Camp David Peace Accords , which have become a national
consensus , supported by 85 % of Israelis according to a 2001 poll taken by the Jaffee Center for Strategic Studies ( Israel based ) .
</item 7>

<item 8>
Camp David Accords
Begin initiative The key to an arrangement between Begin and Sadat took place on Sunday , 6 August 1978 , as a result of a telephone call made that morning to
the Israeli Prime Minister 's office by a United States citizen who had an `` idea for peace . '' The Prime Minister had not yet arrived at his office and the
caller spoke to Mr. Yechiel Kadishai , a Begin staff head . Kadishai said that `` no one was speaking with anyone and we expect a war in October . '' He also
told the caller that if any high level talks were to occur the caller could be assured that they would be using his approach . Begin arrived , was informed of
the plan , and contacted Sadat who agreed to the plan on that day . On the next day , Secretary of State Cyrus Vance traveled to the Middle East to obtain
firsthand confirmation of the agreement between Israel and Egypt . The following day , Tuesday , 8 August , the Camp David meeting was scheduled to take place
in exactly 4 weeks time ; on 5 September 1978 . The plan was that Israel agreed on 6 August to return the land to Egypt .
</item 8>

<item 9>
Camp David Accords
Hussein consequently felt diplomatically snubbed . One of Carter 's regrets was allowing Sadat to claim that he could speak for Hussein if Jordan refused to
join the talks , but by then the damage was done to the Jordanians . United States President Jimmy Carter greeting Egyptian President Anwar Sadat at the White
House shortly after the Camp David Accords went into effect , 8 April 1980 . The Camp David Accords also prompted the disintegration of a united Arab front in
opposition to Israel . Egypt 's realignment created a power vacuum that Saddam Hussein of Iraq , at one time only a secondary power , hoped to fill . Because of
the vague language concerning the implementation of Resolution 242 , the Palestinian problem became the primary issue in the Arab -- Israeli conflict
immediately following the Camp David Accords ( and , arguably , until today ) . Many of the Arab nations blamed Egypt for not putting enough pressure on Israel
to deal with the Palestinian problem in a way that would be satisfactory to them . Syria also informed Egypt that it would not reconcile with the nation unless
it abandoned the peace agreement with Israel .
</item 9>

<item 10>
Camp David Accords
The New York Times . 9 March 1992 . 15 February 2009 . ^ Jump up to : Forward.com Jump up ^ Bitterlemons.org Jump up ^ Foreignpolicyblogs.com Jump up ^
Peacenow.org Jump up ^ `` Camp David Accords : Jimmy Carter Reflects 25 Years Later '' . Carter Center . 17 September 2003 . Retrieved 1 February 2015 . Jump up
^ Stein 1999 , p. 252 . Jump up ^ `` The Camp David Accords . '' Jimmy Carter Library and Museum . 21 July 2001 . 28 April 2008 . ^ Jump up to : Jimmy Carter
Library , The Framework for Peace in the Middle East Archived 16 December 2013 at the Wayback Machine. , 17 September 1978 Jump up ^ Camp David Accords --
Israeli Ministry of Foreign Affairs Archived 3 September 2011 at the Wayback Machine . ^ Jump up to : Stein , 1999 , p. 254 .
</item 10>
=== END ITEM id=7f9f82ce5208e49a ===
