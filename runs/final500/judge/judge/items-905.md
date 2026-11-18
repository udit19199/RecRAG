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

=== ITEM id=83c0299b32082b23 ===
Gold sentences and facts:
Sentence 0: Important production centres today are the Hallertau in Germany ( more hop - growing area than any other country as of 2006 ) , the Yakima ( Washington ) and Willamette ( Oregon ) valleys , and western Canyon County , Idaho ( including the communities of Parma , Wilder , Greenleaf , and Notus ) . The principal production centres in the UK are in Kent ( which produces Kent Goldings hops ) , Herefordshire , and Worcestershire . Essentially all of the harvested hops are used in beer making .
  Fact 0.0: The Hallertau in Germany has more hop-growing area than any other country as of 2006.
  Fact 0.1: The Hallertau in Germany is an important hop production centre today.
  Fact 0.2: The Yakima valley in Washington is an important hop production centre today.
  Fact 0.3: The Willamette valley in Oregon is an important hop production centre today.
  Fact 0.4: Western Canyon County, Idaho is an important hop production centre today.
  Fact 0.5: Western Canyon County, Idaho includes the communities of Parma, Wilder, Greenleaf, and Notus.
  Fact 0.6: The principal hop production centres in the UK are in Kent.
  Fact 0.7: Kent produces Kent Goldings hops.
  Fact 0.8: The principal hop production centres in the UK are in Herefordshire.
  Fact 0.9: The principal hop production centres in the UK are in Worcestershire.
  Fact 0.10: Essentially all of the harvested hops are used in beer making.
Sentence 1: Hops production is concentrated in moist temperate climates , with much of the world 's production occurring near the 48th parallel north . Hop plants prefer the same soils as potatoes and the leading potato - growing states in the United States are also major hops - producing areas ; however , not all potato - growing areas can produce good hops naturally : soils in the Maritime Provinces of Canada , for example , lack the boron that hops prefer . Historically , hops were not grown in Ireland , but were imported from England . In 1752 more than 500 tons of English hops were imported through Dublin alone .
  Fact 1.0: Hops production is concentrated in moist temperate climates.
  Fact 1.1: Much of the world's hops production occurs near the 48th parallel north.
  Fact 1.2: Hop plants prefer the same soils as potatoes.
  Fact 1.3: The leading potato-growing states in the United States are also major hops-producing areas.
  Fact 1.4: Not all potato-growing areas can produce good hops naturally.
  Fact 1.5: Soils in the Maritime Provinces of Canada lack the boron that hops prefer.
  Fact 1.6: Historically, hops were not grown in Ireland.
  Fact 1.7: Historically, hops were imported into Ireland from England.
  Fact 1.8: In 1752 more than 500 tons of English hops were imported through Dublin alone.

Retrieved items:
<item 1>
Hops
Hops were then grown as far north as Aberdeen , near breweries for infrastructure convenience . According to Thomas Tusser 's 1557 Five Hundred Points of Good
Husbandry : `` The hop for his profit I thus do exalt , It strengtheneth drink and it flavoureth malt ; And being well - brewed long kept it will last , And
drawing abide , if ye draw not too fast . '' In England there were many complaints over the quality of imported hops , the sacks of which were often
contaminated by stalks , sand or straw to increase their weight . As a result , in 1603 , King James I approved an Act of Parliament banning the practice by
which `` the Subjects of this Realm have been of late years abused &c. to the Value of £ 20,000 yearly , besides the Danger of their Healths '' . Hop
cultivation was begun in the present - day United States in 1629 by English and Dutch farmers . Before prohibition , cultivation was mainly centred around New
York , California , Oregon , and Washington state .
</item 1>

<item 2>
Hops
Essentially all of the harvested hops are used in beer making . Early season hop growth in a hop yard in the Yakima River Valley of Washington with Mount Adams
in the distance Hop producing country 2012 hop output in tonnes ( t ) Germany 34,434 United States 27,782 China 11,000 Czech Republic 4,338 Poland 2,206 North
Korea 2,000 Albania 1,650 United Kingdom 1,650 Slovenia 1,338 Cultivation and harvest ( edit ) A superstructure of overhead wires supports strings that in turn
support bines Although hops are grown in most of the continental United States and Canada , cultivation of hops for commercial production requires a particular
environment . As hops are a climbing plant , they are trained to grow up trellises made from strings or wires that support the plants and allow them
significantly greater growth with the same sunlight profile . In this way , energy that would have been required to build structural cells is also freed for
crop growth . The hop plant 's reproduction method is that male and female flowers develop on separate plants , although occasionally a fertile individual will
develop which contains both male and female flowers .
</item 2>

<item 3>
Hops
private and public breeding programs are using their stock material . Particular hop varieties are associated with beer regions and styles , for example pale
lagers are usually brewed with European ( often German , Polish or Czech ) noble hop varieties such as Saaz , Hallertau and Strissel Spalt . British ales use
hop varieties such as Fuggles , Goldings and W.G.V. North American beers often use Cascade hops , Columbus hops , Centennial hops , Willamette , Amarillo hops
and about forty more varieties as the U.S.A. have lately been the more significant breeders of new hop varieties , including dwarf hop varieties . Hops from New
Zealand , such as Pacific Gem , Motueka and Nelson Sauvin , are used in a `` Pacific Pale Ale '' style of beer with increasing production in 2014 . Noble hops (
edit ) Mature hops growing in a hop yard in Germany The term `` noble hops '' traditionally refers to varieties of hops which are low in bitterness and high in
aroma . They are the European cultivars or races Hallertau , Tettnanger , Spalt , and Saaz .
</item 3>

<item 4>
Hops
The sample at the brewery Moscow Brewing Company . Hops are usually dried in an oast house before they are used in the brewing process . Undried or `` wet ''
hops are sometimes ( since ca. 1990 ) used . The wort ( sugar - rich liquid produced from malt ) is boiled with hops before it is cooled down and yeast is added
, to start fermentation . The effect of hops on the finished beer varies by type and use , though there are two main hop types : bittering and aroma . Bittering
hops have higher concentrations of alpha acids , and are responsible for the large majority of the bitter flavour of a beer . European ( so - called `` noble ''
) hops typically average 5 -- 9 % alpha acids by weight ( AABW ) , and the newer American cultivars typically range from 8 -- 19 % AABW . Aroma hops usually
have a lower concentration of alpha acids ( ~ 5 % ) and are the primary contributors of hop aroma and ( nonbitter ) flavour . Bittering hops are boiled for a
longer period of time , typically 60 -- 90 minutes , to maximize the isomerization of the alpha acids .
</item 4>

<item 5>
Hops
A heating unit on the lower floor is used to dry the hops . When dry , the hops are moved to a press , a sturdy box with a plunger . Two long pieces of burlap
are laid into the hop press at right angles , the hops are poured in and compressed into bales . Hop cones contain different oils , such as lupulin , a
yellowish , waxy substance , an oleoresin , that imparts flavour and aroma to beer . Lupulin contains lupulone and humulone , which possess antibiotic
properties , suppressing bacterial growth favoring brewer 's yeast to grow . After lupulin has been extracted in the brewing process the papery cones are
discarded . Migrant labor and social impact ( edit ) Hops harvest in the Kingdom of Bohemia ( 1898 ) Hops harvest in Skåne , Sweden in 1937 The need for massed
labor at harvest time meant hop - growing had a big social impact . Around the world , the labor - intensive harvesting work involved large numbers of migrant
workers who would travel for the annual hop harvest .
</item 5>

<item 6>
Hops
461 . Jump up ^ Charles W. Bamforth ( 1998 ) . Beer : tap into the art and science of brewing . Plenum Press . p. 245 . ISBN 0 - 306 - 45797 - 0 . Jump up ^
Knight , Paul D. `` HOPS IN BEER '' . USA Hops . Hop Growers of America . Retrieved 11 June 2015 . Jump up ^ Annual Report of the Commissioner of Patents , Part
2 ; Part 4 . US Patent office . 1858 . p. 291 . ^ Jump up to : Peicanada Jump up ^ `` The London magazine , 1752 '' , page 332 Jump up ^ Summary of Reports :
Nürnberg , Germany , 14 November 2006 , International Hop Growers ' Convention : Economic Committee Jump up ^ `` NCGR - Corvallis Humulus Genetic Resources '' .
Ars-grin.gov . Retrieved 2012 - 05 - 20 . Jump up ^ Norman Moss , A Fancy to Worcesters Archived 29 September 2007 at the Wayback Machine.
</item 6>

<item 7>
Hops
In 1752 more than 500 tons of English hops were imported through Dublin alone . Important production centres today are the Hallertau in Germany ( more hop -
growing area than any other country as of 2006 ) , the Yakima ( Washington ) and Willamette ( Oregon ) valleys , and western Canyon County , Idaho ( including
the communities of Parma , Wilder , Greenleaf , and Notus ) . The principal production centres in the UK are in Kent ( which produces Kent Goldings hops ) ,
Herefordshire , and Worcestershire . Essentially all of the harvested hops are used in beer making .
</item 7>

<item 8>
Hops
Hops - wikipedia Hops Jump to : navigation , search For other uses , see hop ( disambiguation ) . Hop flower in a hop yard in the Hallertau , Germany
Cross-section drawing of a hop Hops are the flowers ( also called seed cones or strobiles ) of the hop plant Humulus lupulus . They are used primarily as a
flavouring and stability agent in beer , to which they impart bitter , zesty , or citric flavours ; though they are also used for various purposes in other
beverages and herbal medicine . The hop plant is a vigorous , climbing , herbaceous perennial , usually trained to grow up strings in a field called a hopfield
, hop garden ( nomenclature in the South of England ) , or hop yard ( in the West Country and U.S. ) when grown commercially . Many different varieties of hops
are grown by farmers around the world , with different types used for particular styles of beer . The first documented use of hops in beer is from the 9th
century , though Hildegard of Bingen , 300 years later , is often cited as the earliest documented source .
</item 8>

<item 9>
Hops
Problems with powdery mildew and downy mildew devastated New York 's production by the 1920s , and California only produces hops on a small scale . Hop bars
were used before modern machinery was invented to make the holes for the hop poles . World production ( edit ) Hops production is concentrated in moist
temperate climates , with much of the world 's production occurring near the 48th parallel north . Hop plants prefer the same soils as potatoes and the leading
potato - growing states in the United States are also major hops - producing areas ; however , not all potato - growing areas can produce good hops naturally :
soils in the Maritime Provinces of Canada , for example , lack the boron that hops prefer . Historically , hops were not grown in Ireland , but were imported
from England . In 1752 more than 500 tons of English hops were imported through Dublin alone .
</item 9>

<item 10>
Hops
They often have inferior aromatic properties , as the aromatic compounds evaporate off during the boil . The degree of bitterness imparted by hops depends on
the degree to which alpha acids are isomerized during the boil , and the impact of a given amount of hops is specified in International Bitterness Units .
Unboiled hops are only mildly bitter . On the other hand , the nonbitter flavour and aroma of hops come from the essential oils , which evaporate during the
boil . Aroma hops are typically added to the wort later to prevent the evaporation of the essential oils , to impart `` hop taste '' ( if during the final 30
minutes of boil ) or `` hop aroma '' ( if during the final 10 minutes , or less , of boil ) . Aroma hops are often added after the wort has cooled and while the
beer ferments , a technique known as `` dry hopping '' , which contributes to the hop aroma . Farnesene is a major component in some hops . The composition of
hop essential oils can differ between varieties and between years in the same variety , having a significant influence on flavour and aroma .
</item 10>
=== END ITEM id=83c0299b32082b23 ===

=== ITEM id=27fcf0867e71efd2 ===
Gold sentences and facts:
Sentence 0: The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016 . Unlike the first six seasons that each had ten episodes and the seventh that had seven episodes , the eighth season will have only six episodes . Like the previous season , it will largely consist of original content not found currently in George R.R. Martin 's A Song of Ice and Fire series , and will instead adapt material Martin has revealed to showrunners about the upcoming novels in the series , The Winds of Winter and A Dream of Spring .
  Fact 0.0: The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016.
  Fact 0.1: The first six seasons of Game of Thrones each had ten episodes.
  Fact 0.2: The seventh season of Game of Thrones had seven episodes.
  Fact 0.3: The eighth season of Game of Thrones will have only six episodes.
  Fact 0.4: Like the previous season, the eighth season will largely consist of original content not found currently in George R.R. Martin's A Song of Ice and Fire series.
  Fact 0.5: The eighth season will instead adapt material Martin has revealed to showrunners about the upcoming novels in the series.
  Fact 0.6: The upcoming novels in the series are The Winds of Winter and A Dream of Spring.

Retrieved items:
<item 1>
Game of Thrones (season 8)
Game of Thrones ( season 8 ) - wikipedia Game of Thrones ( season 8 ) Jump to : navigation , search Game of Thrones ( season 8 ) Promotional poster Starring See
List of Game of Thrones cast Country of origin United States Release Original network HBO Season chronology ← Previous Season 7 List of Game of Thrones episodes
The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016 . Unlike the first six seasons that each
had ten episodes and the seventh that had seven episodes , the eighth season will have only six episodes . Like the previous season , it will largely consist of
original content not found currently in George R.R. Martin 's A Song of Ice and Fire series , and will instead adapt material Martin has revealed to showrunners
about the upcoming novels in the series , The Winds of Winter and A Dream of Spring . The season will be adapted for television by David Benioff and D.B. Weiss
. Filming officially began on October 23 , 2017 . The season is scheduled to premiere in 2019 .
</item 1>

<item 2>
Game of Thrones (season 8)
`` ' Game of Thrones ' Creators Mull Shorter Final Seasons ( EXCLUSIVE ) '' . Variety . Archived from the original on August 21 , 2016 . Retrieved June 27 ,
2016 . Jump up ^ Goldberg , Lesley ( April 14 , 2016 ) . `` ' Game of Thrones ' Considering Shorter Final Seasons '' . The Hollywood Reporter . Archived from
the original on June 18 , 2016 . Retrieved June 27 , 2016 . Jump up ^ Hibberd , James ( July 18 , 2016 ) . `` Game of Thrones : HBO announces summer return , 7
episodes '' . Entertainment Weekly . Archived from the original on September 1 , 2016 . Retrieved July 18 , 2016 . ^ Jump up to : Hibberd , James ( March 12 ,
2017 ) . `` Game of Thrones producers confirm final season only 6 episodes '' . Entertainment Weekly . Archived from the original on March 12 , 2017 . Retrieved
March 13 , 2017 .
</item 2>

<item 3>
Game of Thrones (season 8)
The final episode of the show will be directed by Benioff and Weiss , who have previously directed one episode each . At the show 's South by Southwest panel on
March 12 , 2017 , Benioff and Weiss announced the writers for the show to be Dave Hill ( episode 1 ) and Bryan Cogman ( episode 2 ) . The showrunners will then
divide up the screenplay for the remaining four episodes amongst themselves . Writing ( edit ) Writing for the eighth season started with a 140 - page outline .
Benioff said that the divvying up process and who should write what section became more difficult , being that `` this would be the last time that we would be
doing this . '' Filming ( edit ) In an interview with Entertainment Weekly , HBO programming president Casey Bloys stated that instead of the series finale
being a feature film , the final season would be `` six one - hour movies '' on television . He continued , `` The show has proven that TV is every bit as
impressive and in many cases more so , than film . What they 're doing is monumental . ''
</item 3>

<item 4>
Game of Thrones (season 8)
What they 're doing is monumental . '' Filming officially began on October 23 , 2017 . Episodes ( edit ) Co-creators David Benioff and D.B. Weiss have said that
the seventh and eighth season would likely consist of fewer episodes , stating that after season six , they were `` down to our final 13 episodes after this
season . We 're heading into the final lap . '' Benioff and Weiss stated that they were unable to produce 10 episodes in the show 's usual 12 to 14 month time
frame , as Weiss said , `` It 's crossing out of a television schedule into more of a mid-range movie schedule . '' HBO confirmed in July 2016 , that the
seventh season would consist of seven episodes , and would premiere later than usual in mid-2017 because of the later filming schedule . Benioff and Weiss later
confirmed that the eighth season will consist of six episodes , and is expected to premiere later than usual for the same reason .
</item 4>

<item 5>
Game of Thrones (season 8)
Benioff and Weiss spoke about the end of the show , saying , `` From the beginning we 've wanted to tell a 70 - hour movie . It will turn out to be a 73 - hour
movie , but it 's stayed relatively the same of having the beginning , middle and now we 're coming to the end . It would have been really tough if we lost any
core cast members along the way , I 'm very happy we 've kept everyone and we get to finish it the way we want to . '' The season is set to air in 2019 . Music
( edit ) Ramin Djawadi is set to return as the composer of the show for the eighth season . References ( edit ) Jump up ^ Hibberd , James ( July 30 , 2016 ) .
`` Game of Thrones : HBO confirms season 8 will be last '' . Entertainment Weekly . Archived from the original on September 1 , 2016 . Retrieved March 13 , 2017
. Jump up ^ Birnbaum , Debra ( July 30 , 2016 ) .
</item 5>

<item 6>
Game of Thrones (season 8)
Queen ( 2013 ) The Rogue Prince ( 2014 ) The Sons of the Dragon ( 2017 ) TV series Episodes Season 1 5 6 7 8 Awards Characters Music `` Game of Thrones Theme ''
`` The Rains of Castamere '' `` The Bear and the Maiden Fair '' `` Light of the Seven '' Catch the Throne Live Concert Experience Soundtracks Season 1 5 6 7
Title sequence After the Thrones Thronecast Video games A Game of Thrones : Genesis Game of Thrones ( 2012 ) Game of Thrones Ascent Game of Thrones ( 2014 )
Other media The World of Ice & Fire ( 2014 ) Board game Card game ( Second edition ) Comic book series Characters Petyr Baelish Joffrey Baratheon Renly
Baratheon Robert Baratheon Stannis Baratheon Tommen Baratheon Ramsay Bolton Roose Bolton Bronn Gregor Clegane Sandor Clegane Khal Drogo Gendry Tormund
Giantsbane Gilly Theon Greyjoy Cersei Lannister Jaime Lannister Tyrion Lannister Tywin Lannister Oberyn Martell Melisandre Missandei Jorah
</item 6>

<item 7>
Game of Thrones (season 8)
The season is scheduled to premiere in 2019 . Contents ( hide ) 1 Episodes 2 Cast 2.1 Main cast 2.2 Supporting cast 3 Production 3.1 Crew 3.2 Writing 3.3
Filming 3.4 Episodes 3.5 Music 4 References 5 External links Episodes ( edit ) See also : List of Game of Thrones episodes No . overall No. in season Title
Directed by Written by Original air date U.S. viewers ( millions ) 68 TBA David Nutter Dave Hill 2019 ( 2019 ) TBD 69 TBA David Nutter Bryan Cogman 2019 ( 2019
) TBD 70 TBA Miguel Sapochnik David Benioff & D.B. Weiss 2019 ( 2019 ) TBD 71 TBA David Nutter David Benioff & D.B. Weiss 2019 ( 2019 ) TBD 72 5 TBA Miguel
Sapochnik David Benioff & D.B.
</item 7>

<item 8>
Game of Thrones (season 8)
Jump up ^ Chitwood , Adam ( December 15 , 2017 ) . `` ' Justice League ' DP Fabian Wagner on Zack Snyder 's Cut , Superman 's Black Suit & ' Game of Thrones '
'' . Collider . Archived from the original on December 16 , 2017 . Retrieved December 18 , 2017 . Jump up ^ Carson , Biz ( March 12 , 2017 ) . `` The final
season of ' Game of Thrones ' will only be six episodes '' . Business Insider . Archived from the original on April 12 , 2017 . Retrieved April 11 , 2017 . ^
Jump up to : `` Game of Thrones Will Return in 2019 '' . HBO . Archived from the original on January 4 , 2018 . Retrieved January 4 , 2018 . ^ Jump up to :
Goldberg , Lesley ( June 21 , 2016 ) . `` ' Game of Thrones ' Stars Score Hefty Pay Raises for Season 8 '' . The Hollywood Reporter .
</item 8>

<item 9>
Game of Thrones (season 8)
Weiss 2019 ( 2019 ) TBD 73 6 TBA David Benioff & D.B. Weiss David Benioff & D.B. Weiss 2019 ( 2019 ) TBD Cast ( edit ) See also : List of Game of Thrones
characters Peter Dinklage ( Tyrion Lannister ) Nikolaj Coster - Waldau ( Jaime Lannister ) Lena Headey ( Cersei Lannister ) Emilia Clarke ( Daenerys Targaryen )
Main cast ( edit ) Peter Dinklage as Tyrion Lannister Nikolaj Coster - Waldau as Jaime Lannister Lena Headey as Cersei Lannister Emilia Clarke as Daenerys
Targaryen Kit Harington as Jon Snow Liam Cunningham as Davos Seaworth Sophie Turner as Sansa Stark Maisie Williams as Arya Stark Nathalie Emmanuel as Missandei
Gwendoline Christie as Brienne of Tarth John Bradley as Samwell Tarly Isaac Hempstead Wright as Bran Stark Rory McCann as Sandor `` The Hound '' Clegane Iain
Glen as Jorah Mormont Conleth Hill as Varys
</item 9>

<item 10>
Game of Thrones (season 8)
Jump up ^ Birnbaum , Debra ( July 30 , 2016 ) . `` HBO Confirms ' Game of Thrones ' Will End With Eighth Season '' . Variety . Archived from the original on
September 2 , 2016 . Retrieved March 13 , 2017 . Jump up ^ Hibberd , James ( May 24 , 2016 ) . `` George R.R. Martin revealed 3 huge shocks to Game of Thrones
producers '' . Entertainment Weekly . Archived from the original on August 21 , 2016 . Retrieved March 13 , 2017 . ^ Jump up to : Hooton , Christopher (
September 27 , 2017 ) . `` Game of Thrones season 8 : Directors revealed for all episodes '' . The Independent . Archived from the original on October 20 , 2017
. Retrieved October 20 , 2017 . Jump up ^ Chitwood , Adam ( December 15 , 2017 ) .
</item 10>
=== END ITEM id=27fcf0867e71efd2 ===

=== ITEM id=a78ee66b3bab002b ===
Gold sentences and facts:
Sentence 0: The following season , a cancerous desmoid tumor was found in Dravecky 's pitching arm . On October 7 , 1988 , he underwent surgery , which removed half of the deltoid muscle in his pitching arm and froze the humerus bone in an effort to eliminate all of the cancerous cells . Doctors advised Dravecky to wait until 1990 to pitch again , but Dravecky was determined to pitch in 1989 . By July 1989 , he was pitching in the minors , and on August 10 , he made a highly publicized return to the major leagues , pitching eight innings and defeating Cincinnati 4 -- 3 . In his following start , five days later in Montreal , Dravecky pitched three no - hit innings , but in the fifth inning , he felt a tingling sensation in his arm . In the sixth inning , he started off shaky , allowing a home run to the leadoff batter and then hitting the second batter , Andrés Galarraga . Then , on his first pitch to Tim Raines , his humerus bone snapped ; the sound of it breaking could be heard throughout the stadium . Dravecky collapsed on the mound . He 'd suffered a clean break midway between his shoulder and elbow , ending his season .
  Fact 0.0: The following season, a cancerous desmoid tumor was found in Dravecky's pitching arm.
  Fact 0.1: On October 7, 1988, Dravecky underwent surgery.
  Fact 0.2: The surgery removed half of the deltoid muscle in Dravecky's pitching arm.
  Fact 0.3: The surgery froze the humerus bone in Dravecky's pitching arm in an effort to eliminate all of the cancerous cells.
  Fact 0.4: Doctors advised Dravecky to wait until 1990 to pitch again.
  Fact 0.5: Dravecky was determined to pitch in 1989.
  Fact 0.6: By July 1989, Dravecky was pitching in the minors.
  Fact 0.7: On August 10, Dravecky made a highly publicized return to the major leagues.
  Fact 0.8: In that return, Dravecky pitched eight innings and defeated Cincinnati 4-3.
  Fact 0.9: In his following start, five days later in Montreal, Dravecky pitched three no-hit innings.
  Fact 0.10: In the fifth inning of that start, Dravecky felt a tingling sensation in his arm.
  Fact 0.11: In the sixth inning, Dravecky started off shaky.
  Fact 0.12: In the sixth inning, Dravecky allowed a home run to the leadoff batter.
  Fact 0.13: In the sixth inning, Dravecky hit the second batter, Andrés Galarraga.
  Fact 0.14: On his first pitch to Tim Raines, Dravecky's humerus bone snapped.
  Fact 0.15: The sound of the bone breaking could be heard throughout the stadium.
  Fact 0.16: Dravecky collapsed on the mound.
  Fact 0.17: Dravecky suffered a clean break midway between his shoulder and elbow.
  Fact 0.18: The broken bone ended Dravecky's season.

Retrieved items:
<item 1>
2005 World Series
Chris Widger 38 Pablo Ozuna 43 Dámaso Marte 45 Bobby Jenks 46 Neal Cotts 51 Luis Vizcaíno 52 José Contreras 56 Mark Buehrle Manager 13 Ozzie Guillén Bench Coach
3 Harold Baines First Base Coach 30 Tim Raines Third Base Coach 28 Joey Cora Pitching Coach 21 Don Cooper Hitting Coach 29 Greg Walker Bullpen Coach 53 Art
Kusnyer Bullpen Catcher 59 Man - soo Lee General Manager Kenny Williams Regular season American League Division Series American League Championship Series 2005
MLB season by team AL East Baltimore Boston New York Tampa Bay Toronto AL Central Chicago Cleveland Detroit Kansas City Minnesota AL West Los Angeles Oakland
Seattle Texas NL East Atlanta Florida New York Philadelphia Washington NL Central Chicago Cincinnati Houston Milwaukee Pittsburgh St.
</item 1>

<item 2>
2017 World Series
Game 1 : The ceremonial first pitch was thrown out by members of former Dodger Jackie Robinson 's family , including his widow Rachel . The game marked the 45th
anniversary of Robinson 's death , and the 2017 season was the 70th anniversary of his breaking of the baseball color line . Keith Williams Jr. , a gospel
singer , performed `` The Star - Spangled Banner '' , the national anthem . Game 2 : Fernando Valenzuela threw out the ceremonial first pitch to Steve Yeager ;
both were introduced by retired Dodgers broadcaster Vin Scully . Country music 's Brad Paisley performed the national anthem . Game 3 : Houston Texans defensive
end J.J. Watt , who had raised $37 million for Hurricane Harvey victims , threw out the first pitch . The ball was given to him by Astros ' Hall of Famer Craig
Biggio . The national anthem was performed by Texas Air National Guard Master Sergeant Promise Harris . Game 4 : Hailey Dawson , a seven - year - old girl from
Nevada , threw out the ceremonial first pitch using a 3D printed hand . The national anthem was performed by the Houston Police Department Quartet .
</item 2>

<item 3>
2017 World Series
J. Pierzynski Harold Reynolds Frank Robinson Ken Singleton John Smoltz Jeff Torborg Tom Verducci Guest commentators Bret Boone David Cone Terry Francona Luis
Gonzalez Ozzie Guillén Al Leiter David Ortiz A.J. Pierzynski Lou Piniella Jimmy Rollins Nick Swisher Field reporters Erin Andrews Curt Menefee Chris Myers Ken
Rosenthal Studio hosts Greg Amsinger Kevin Burkhardt Chip Caray Brian Kenny Keith Olbermann Patrick O'Neal Chris Rose Rob Stone Jeanne Zelasko Studio analysts
Eric Byrnes Keith Hernandez Raúl Ibañez Gabe Kapler Kevin Kennedy Kevin Millar C.J. Nitkowski Dan Plesac Billy Ripken Pete Rose Mark Sweeney Frank Thomas Mitch
Williams Dontrelle Willis Dave Winfield Lore The Flip Play ( 2001 ) Steve Bartman ( 2003 ) Yankees -- Red Sox rivalry The 53 - Minute 7th Inning ( 2015 )
Regular season 1998 Major League Baseball home run record chase Philip Humber 's perfect game ( 2012 ) World Series games The Last Night of the Yankee Dynasty (
2001 ) Game 6 of the 2011 World Series Walk -
</item 3>

<item 4>
2005 World Series
J. Pierzynski Harold Reynolds Frank Robinson Ken Singleton John Smoltz Jeff Torborg Tom Verducci Guest commentators Bret Boone David Cone Terry Francona Luis
Gonzalez Ozzie Guillén Al Leiter David Ortiz A.J. Pierzynski Lou Piniella Jimmy Rollins Nick Swisher Field reporters Erin Andrews Curt Menefee Chris Myers Ken
Rosenthal Studio hosts Greg Amsinger Kevin Burkhardt Chip Caray Brian Kenny Keith Olbermann Patrick O'Neal Chris Rose Rob Stone Jeanne Zelasko Studio analysts
Eric Byrnes Keith Hernandez Raúl Ibañez Gabe Kapler Kevin Kennedy Kevin Millar C.J. Nitkowski Dan Plesac Billy Ripken Pete Rose Mark Sweeney Frank Thomas Mitch
Williams Dontrelle Willis Dave Winfield Lore The Flip Play ( 2001 ) Steve Bartman ( 2003 ) Yankees -- Red Sox rivalry The 53 - Minute 7th Inning ( 2015 )
Regular season 1998 Major League Baseball home run record chase Philip Humber 's perfect game ( 2012 ) World Series games The Last Night of the Yankee Dynasty (
2001 ) Game 6 of the 2011 World Series Walk -
</item 4>

<item 5>
Dave Dravecky
Crisis and comeback ( edit ) The following season , a cancerous desmoid tumor was found in Dravecky 's pitching arm . On October 7 , 1988 , he underwent surgery
, which removed half of the deltoid muscle in his pitching arm and froze the humerus bone in an effort to eliminate all of the cancerous cells . Doctors advised
Dravecky to wait until 1990 to pitch again , but Dravecky was determined to pitch in 1989 . By July 1989 , he was pitching in the minors , and on August 10 , he
made a highly publicized return to the major leagues , pitching eight innings and defeating Cincinnati 4 -- 3 . In his following start , five days later in
Montreal , Dravecky pitched three no - hit innings , but in the fifth inning , he felt a tingling sensation in his arm . In the sixth inning , he started off
shaky , allowing a home run to the leadoff batter and then hitting the second batter , Andrés Galarraga .
</item 5>

<item 6>
Dave Dravecky
Fred Anderson Johnny Antonelli Jesse Barnes Virgil Barnes Jim Barr Larry Benton Vida Blue Tom Bradley Madison Bumgarner John Burkett Matt Cain Mark Davis Phil
Douglas Dave Dravecky Shawn Estes Freddie Fitzsimmons Mark Gardner Rubén Gómez Harry Gumbert Atlee Hammaker Liván Hernández Al Holland Carl Hubbell Larry Jansen
Bob Knepper Mike Krukow Mark Leiter Tim Lincecum Bill Lohrman Sal Maglie Juan Marichal Rube Marquard Christy Mathewson Joe McGinnity Hugh McQuillan Cliff Melton
John Montefusco Terry Mulholland Art Nehf Gaylord Perry Rick Reuschel Kirk Rueter Rosy Ryan Jack Sanford Jason Schmidt Hal Schumacher Bill Swift Dummy Taylor
Jeff Tesreau Bill Voiselle Bill Walker Barry Zito VIAF : 77929919 LCCN : n89666209 Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Dave Dravecky
Then , on his first pitch to Tim Raines , his humerus bone snapped ; the sound of it breaking could be heard throughout the stadium . Dravecky collapsed on the
mound . He 'd suffered a clean break midway between his shoulder and elbow , ending his season . The Giants won the National League pennant in 1989 ( defeating
the Chicago Cubs in the NLCS in five games ) , and in the post-game celebration , Dravecky 's arm was broken a second time when he was running out to the mound
to celebrate . A doctor examining Dravecky 's x-rays noticed a mass in his arm , which turned out to be malignant . Dravecky 's cancer had returned , ending his
career . Eighteen days later , Dravecky retired from baseball with a 64 -- 57 record with 558 strikeouts and a 3.13 ERA in 1,062 2⁄3 innings . He won the 1989
Willie Mac Award honoring his spirit and leadership .
</item 7>

<item 8>
Dave Dravecky
1980 : Brett 1981 : Bench 1982 : Thornton 1983 : Knight 1984 : Robinson 1985 : Reuschel 1986 : Leonard 1987 : Molitor 1988 : Oester 1989 : Dravecky 1990 : Bream
1991 : Wegman 1992 : Lansford 1993 : Olerud 1994 : Dawson 1995 : Abbott 1996 : Vizquel 1997 : Davis 1998 : Cone 1999 : Casey 2000 : Giambi 2001 : Schilling 2002
: Salmon 2003 : Moyer 2004 : Hoffman 2005 : Biggio 2006 : Loretta 2007 : Sweeney 2008 : Lester 2009 : Teahen 2010 : Hudson 2011 : Butler 2012 : Zito 2013 :
Ibañez 2014 : Gordon 2015 : Wainwright 2016 : McGowan New York / San Francisco Giants Opening Day starting pitchers Red Ames Fred Anderson Johnny Antonelli
Jesse Barnes Virgil Barnes Jim Barr Larry Benton Vida Blue Tom Bradley Madison
</item 8>

<item 9>
Houston Astros
Surgery was done to save his life , but the Astros had lost their ace pitcher after a 10 -- 4 start with a stingy 1.89 ERA . Richard attempted a comeback , but
would never again pitch a big league game . After the loss of Richard and some offensive struggles , the Astros slipped to third place in the division behind
the Dodgers and the Reds . They bounced back to first with a ten - game winning streak , but the Dodgers had regained a two - game lead when they arrived in
Houston on September 9 . The Astros won the first two games of that series and the two teams were tied for the division lead . The Astros held a three - game
lead over the Dodgers with three games left in the season against the Dodgers . The Dodgers swept the series games , forcing a one - game playoff the next day .
The Astros would however win the one - game playoff 7 - 1 , and advance to their first post-season . The team would face the Philadelphia Phillies in the 1980
National League Championship Series . The Phillies sent out Steve Carlton in game one of the NLCS after a six - hour flight the night before .
</item 9>

<item 10>
2005 World Series
2005 World Series - wikipedia 2005 World Series Jump to : navigation , search 2005 World Series Team ( Wins ) Manager ( s ) Season Chicago White Sox ( 4 ) Ozzie
Guillén 99 -- 63 , . 611 , GA : 6 Houston Astros ( 0 ) Phil Garner 89 -- 73 , . 549 , GB : 11 Dates October 22 -- 26 MVP Jermaine Dye ( Chicago ) Umpires Joe
West ( crew chief ) , Jeff Nelson , Jerry Layne , Derryl Cousins , Gary Cederstrom , Ángel Hernández Hall of Famers White Sox : Tim Raines ( 1B Coach ) , Frank
Thomas ( DNP ) Astros : Jeff Bagwell , Craig Biggio ALCS Chicago White Sox defeated Los Angeles Angels of Anaheim , 4 -- 1 NLCS Houston Astros defeated St.
</item 10>
=== END ITEM id=a78ee66b3bab002b ===

=== ITEM id=50c9bd1e1cc2da58 ===
Gold sentences and facts:
Sentence 0: No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 ) Labour 27 April 1904 18 August 1904 Bland , NSW , 1901 -- 1906 South Sydney , NSW , 1906 -- 1910 ( retired ) -- Watson George Reid ( 1845 -- 1918 ) Free Trade 18 August 1904 5 July 1905 East Sydney , NSW , 1901 -- 1909 ( resigned ) -- Reid ( 2 ) Alfred Deakin ( 1856 -- 1919 ) Protectionist 5 July 1905 13 November 1908 Ballaarat , Vic , 1901 -- 1913 ( retired ) -- 2nd Deakin 1906 3rd Deakin 5 Andrew Fisher ( 1862 -- 1928 ) Labour 13 November 1908 2 June 1909 Wide Bay , Qld , 1901 -- 1915 ( resigned ) -- 1st Fisher ( 2 ) Alfred Deakin ( 1856 -- 1919 ) Commonwealth Liberal 2 June 1909 29 April 1910 Ballaarat , Vic , 1901 -- 1913 ( retired ) -- 4th Deakin ( 5 ) Andrew Fisher ( 1862 -- 1928 ) Labor 29 April 1910 24 June 1913 Wide Bay , Qld , 1901 -- 1915 ( resigned ) 1910 2nd Fisher 6 Joseph Cook ( 1860 -- 1947 ) Commonwealth Liberal 24 June 1913 17 September 1914 Parramatta , NSW , 1901 -- 1921 ( resigned ) 1913 Cook ( 5 ) Andrew Fisher ( 1862 -- 1928 ) Labor 17 September 1914 27 October 1915 Wide Bay , Qld , 1901 -- 1915 ( resigned ) 1914 3rd Fisher Billy Hughes ( 1862 -- 1952 ) Labor 27 October 1915 14 November 1916 West Sydney , NSW , 1901 -- 1917 Bendigo , Vic , 1917 -- 1922 North Sydney , NSW , 1922 -- 1949 Bradfield , NSW , 1949 -- 1952 ( died ) -- 1st Hughes 7 National Labor 14 November 1916 17 February 1917 -- 2nd Hughes Nationalist 17 February 1917 9 February 1923 -- 3rd Hughes 1917 4th Hughes 1919 5th Hughes 8 Stanley Bruce ( 1883 -- 1967 ) Nationalist ( Coalition ) 9 February 1923 22 October 1929 Flinders , Vic , 1918 -- 1929 ( defeated ) ; 1931 -- 1933 ( resigned ) 1922 1st Bruce 1925 2nd Bruce 1928 3rd Bruce 9 James Scullin ( 1876 -- 1953 ) Labor 22 October 1929 6 January 1932 Corangamite , Vic , 1910 -- 1913 ( defeated ) Yarra , Vic , 1922 -- 1949 ( retired ) 1929 Scullin 10 Joseph Lyons ( 1879 -- 1939 ) United Australia ( Coalition ) 6 January 1932 7 April 1939 † Wilmot , Tas , 1929 -- 1939 ( died ) 1931 1st Lyons 1934 2nd Lyons -- 3rd Lyons 1937 4th Lyons 11 Sir Earle Page ( 1880 -- 1961 ) Country ( Coalition ) 7 April 1939 26 April 1939 Cowper , NSW 1919 -- 1961 ( defeated ) -- Page 12 Robert Menzies ( 1894 -- 1978 ) United Australia ( Coalition ) 26 April 1939 28 August 1941 Kooyong , Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October 1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 † Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945 Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW , 1928 -- 1931 ( defeated ) ; 1940 -- 1951 ( died ) -- 1st Chifley 1946 2nd Chifley ( 12 ) Sir Robert Menzies ( 1894 -- 1978 ) Liberal ( Coalition ) 19 December 1949 26 January 1966 Kooyong , Vic , 1934 -- 1966 ( resigned ) 1949 4th Menzies 1951 5th Menzies 1954 6th Menzies 1955 7th Menzies 1958 8th Menzies 1961 9th Menzies 1963 10th Menzies 17 Harold Holt ( 1908 -- 1967 ) Liberal ( Coalition ) 26 January 1966 19 December 1967 † Fawkner , Vic , 1935 -- 1949 Higgins , Vic , 1949 -- 1967 ( disappeared ) -- 1st Holt 1966 2nd Holt 18 John McEwen ( 1900 -- 1980 ) Country ( Coalition ) 19 December 1967 10 January 1968 Echuca , Vic , 1934 -- 1937 Indi , Vic , 1937 -- 1949 Murray , Vic , 1949 -- 1971 ( resigned ) -- McEwen 19 John Gorton ( 1911 -- 2002 ) Liberal ( Coalition ) 10 January 1968 10 March 1971 Senator 1950 -- 1968 ( resigned ) MP for Higgins , Vic , 1968 -- 1975 ( retired ) -- 1st Gorton 1969 2nd Gorton 20 William McMahon ( 1908 -- 1988 ) Liberal ( Coalition ) 10 March 1971 5 December 1972 Lowe , NSW , 1949 -- 1982 ( resigned ) -- McMahon 21 Gough Whitlam ( 1916 -- 2014 ) Labor 5 December 1972 11 November Werriwa , NSW , 1952 -- 1978 ( resigned ) 1972 1st Whitlam -- 2nd Whitlam 3rd Whitlam 22 Malcolm Fraser ( 1930 -- 2015 ) Liberal ( Coalition ) 11 November 11 March Wannon , Vic , 1955 -- 1983 ( resigned ) -- 1st Fraser 2nd Fraser 1977 3rd Fraser 1980 4th Fraser 23 Bob Hawke ( 1929 -- ) Labor 11 March 20 December 1991 Wills , Vic , 1980 -- 1992 ( resigned ) 1st Hawke 1984 2nd Hawke 3rd Hawke 1990 4th Hawke 24 Paul Keating ( 1944 -- ) Labor 20 December 1991 11 March Blaxland , NSW , 1969 -- 1996 ( resigned ) -- 1st Keating 1993 2nd Keating 25 John Howard ( 1939 -- ) Liberal ( Coalition ) 11 March 3 December 2007 Bennelong , NSW , 1974 -- 2007 ( defeated ) 1st Howard 1998 2nd Howard 2001 3rd Howard 4th Howard 26 Kevin Rudd ( 1957 -- ) Labor 3 December 2007 24 June Griffith , Qld , 1998 -- 2013 ( resigned ) 2007 1st Rudd 27 Julia Gillard ( 1961 -- ) Labor 24 June 27 June 2013 Lalor , Vic , 1998 -- 2013 ( retired ) -- 1st Gillard 2nd Gillard ( 26 ) Kevin Rudd ( 1957 -- ) Labor 27 June 2013 18 September 2013 Griffith , Qld , 1998 -- 2013 ( resigned ) -- 2nd Rudd 28 Tony Abbott ( 1957 -- ) Liberal ( Coalition ) 18 September 2013 15 September 2015 Warringah , NSW , since 1994 2013 Abbott 29 Malcolm Turnbull ( 1954 -- ) Liberal ( Coalition ) 15 September 2015 Incumbent Wentworth , NSW , since 2004 -- 1st Turnbull 2016 2nd Turnbull
  Fact 0.0: Sir Edmund Barton (1849 to 1920) was Prime Minister from 1 January 1901 to 24 September 1903.
  Fact 0.1: Edmund Barton was a Protectionist.
  Fact 0.2: Edmund Barton represented Hunter, NSW, from 1901 to 1903 and resigned.
  Fact 0.3: Edmund Barton won the 1901 election.
  Fact 0.4: Edmund Barton led the Barton ministry.
  Fact 0.5: Alfred Deakin (1856 to 1919) was Prime Minister from 24 September 1903 to 27 April 1904.
  Fact 0.6: Alfred Deakin was a Protectionist.
  Fact 0.7: Alfred Deakin represented Ballaarat, Vic, from 1901 to 1913 and retired.
  Fact 0.8: Alfred Deakin won the 1903 election.
  Fact 0.9: Alfred Deakin led the 1st Deakin ministry.
  Fact 0.10: Chris Watson (1867 to 1941) was Prime Minister from 27 April 1904 to 18 August 1904.
  Fact 0.11: Chris Watson was a Labour member.
  Fact 0.12: Chris Watson represented Bland, NSW, from 1901 to 1906.
  Fact 0.13: Chris Watson represented South Sydney, NSW, from 1906 to 1910 and retired.
  Fact 0.14: Chris Watson led the Watson ministry.
  Fact 0.15: George Reid (1845 to 1918) was Prime Minister from 18 August 1904 to 5 July 1905.
  Fact 0.16: George Reid was a Free Trade member.
  Fact 0.17: George Reid represented East Sydney, NSW, from 1901 to 1909 and resigned.
  Fact 0.18: George Reid led the Reid ministry.
  Fact 0.19: Alfred Deakin was Prime Minister again from 5 July 1905 to 13 November 1908.
  Fact 0.20: Alfred Deakin was a Protectionist in this term.
  Fact 0.21: Alfred Deakin led the 2nd Deakin ministry.
  Fact 0.22: Alfred Deakin won the 1906 election.
  Fact 0.23: Alfred Deakin led the 3rd Deakin ministry after the 1906 election.
  Fact 0.24: Andrew Fisher (1862 to 1928) was Prime Minister from 13 November 1908 to 2 June 1909.
  Fact 0.25: Andrew Fisher was a Labour member.
  Fact 0.26: Andrew Fisher represented Wide Bay, Qld, from 1901 to 1915 and resigned.
  Fact 0.27: Andrew Fisher led the 1st Fisher ministry.
  Fact 0.28: Alfred Deakin was Prime Minister from 2 June 1909 to 29 April 1910.
  Fact 0.29: Alfred Deakin was a Commonwealth Liberal in this term.
  Fact 0.30: Alfred Deakin led the 4th Deakin ministry.
  Fact 0.31: Andrew Fisher was Prime Minister from 29 April 1910 to 24 June 1913.
  Fact 0.32: Andrew Fisher was a Labor member in this term.
  Fact 0.33: Andrew Fisher won the 1910 election.
  Fact 0.34: Andrew Fisher led the 2nd Fisher ministry.
  Fact 0.35: Joseph Cook (1860 to 1947) was Prime Minister from 24 June 1913 to 17 September 1914.
  Fact 0.36: Joseph Cook was a Commonwealth Liberal.
  Fact 0.37: Joseph Cook represented Parramatta, NSW, from 1901 to 1921 and resigned.
  Fact 0.38: Joseph Cook won the 1913 election.
  Fact 0.39: Joseph Cook led the Cook ministry.
  Fact 0.40: Andrew Fisher was Prime Minister from 17 September 1914 to 27 October 1915.
  Fact 0.41: Andrew Fisher was a Labor member in this term.
  Fact 0.42: Andrew Fisher won the 1914 election.
  Fact 0.43: Andrew Fisher led the 3rd Fisher ministry.
  Fact 0.44: Billy Hughes (1862 to 1952) was Prime Minister from 27 October 1915 to 14 November 1916.
  Fact 0.45: Billy Hughes was a Labor member.
  Fact 0.46: Billy Hughes represented West Sydney, NSW, from 1901 to 1917.
  Fact 0.47: Billy Hughes represented Bendigo, Vic, from 1917 to 1922.
  Fact 0.48: Billy Hughes represented North Sydney, NSW, from 1922 to 1949.
  Fact 0.49: Billy Hughes represented Bradfield, NSW, from 1949 to 1952, ending with his death.
  Fact 0.50: Billy Hughes led the 1st Hughes ministry.
  Fact 0.51: Billy Hughes was Prime Minister from 14 November 1916 to 17 February 1917 as a National Labor member.
  Fact 0.52: Billy Hughes led the 2nd Hughes ministry.
  Fact 0.53: Billy Hughes was Prime Minister from 17 February 1917 to 9 February 1923 as a Nationalist.
  Fact 0.54: Billy Hughes led the 3rd Hughes ministry.
  Fact 0.55: Billy Hughes won the 1917 election and led the 4th Hughes ministry.
  Fact 0.56: Billy Hughes won the 1919 election and led the 5th Hughes ministry.
  Fact 0.57: Stanley Bruce (1883 to 1967) was Prime Minister from 9 February 1923 to 22 October 1929.
  Fact 0.58: Stanley Bruce was a Nationalist (Coalition) member.
  Fact 0.59: Stanley Bruce represented Flinders, Vic, from 1918 to 1929 and was defeated.
  Fact 0.60: Stanley Bruce represented Flinders, Vic, from 1931 to 1933 and resigned.
  Fact 0.61: Stanley Bruce won the 1922 election and led the 1st Bruce ministry.
  Fact 0.62: Stanley Bruce won the 1925 election and led the 2nd Bruce ministry.
  Fact 0.63: Stanley Bruce won the 1928 election and led the 3rd Bruce ministry.
  Fact 0.64: James Scullin (1876 to 1953) was Prime Minister from 22 October 1929 to 6 January 1932.
  Fact 0.65: James Scullin was a Labor member.
  Fact 0.66: James Scullin represented Corangamite, Vic, from 1910 to 1913 and was defeated.
  Fact 0.67: James Scullin represented Yarra, Vic, from 1922 to 1949 and retired.
  Fact 0.68: James Scullin won the 1929 election and led the Scullin ministry.
  Fact 0.69: Joseph Lyons (1879 to 1939) was Prime Minister from 6 January 1932 to 7 April 1939.
  Fact 0.70: Joseph Lyons was a United Australia (Coalition) member.
  Fact 0.71: Joseph Lyons represented Wilmot, Tas, from 1929 to 1939 and died in office.
  Fact 0.72: Joseph Lyons won the 1931 election and led the 1st Lyons ministry.
  Fact 0.73: Joseph Lyons won the 1934 election and led the 2nd Lyons ministry.
  Fact 0.74: Joseph Lyons led the 3rd Lyons ministry.
  Fact 0.75: Joseph Lyons won the 1937 election and led the 4th Lyons ministry.
  Fact 0.76: Sir Earle Page (1880 to 1961) was Prime Minister from 7 April 1939 to 26 April 1939.
  Fact 0.77: Earle Page was a Country (Coalition) member.
  Fact 0.78: Earle Page represented Cowper, NSW, from 1919 to 1961 and was defeated.
  Fact 0.79: Earle Page led the Page ministry.
  Fact 0.80: Robert Menzies (1894 to 1978) was Prime Minister from 26 April 1939 to 28 August 1941.
  Fact 0.81: Robert Menzies was a United Australia (Coalition) member.
  Fact 0.82: Robert Menzies represented Kooyong, Vic, from 1934 to 1966 and resigned.
  Fact 0.83: Robert Menzies led the 1st Menzies ministry.
  Fact 0.84: Robert Menzies led the 2nd Menzies ministry.
  Fact 0.85: Robert Menzies won the 1940 election and led the 3rd Menzies ministry.
  Fact 0.86: Arthur Fadden (1894 to 1973) was Prime Minister from 28 August 1941 to 7 October 1941.
  Fact 0.87: Arthur Fadden was a Country (Coalition) member.
  Fact 0.88: Arthur Fadden represented Darling Downs, Qld, from 1936 to 1949.
  Fact 0.89: Arthur Fadden represented McPherson, Qld, from 1949 to 1958 and retired.
  Fact 0.90: Arthur Fadden led the Fadden ministry.
  Fact 0.91: John Curtin (1885 to 1945) was Prime Minister from 7 October 1941 to 5 July 1945.
  Fact 0.92: John Curtin was a Labor member.
  Fact 0.93: John Curtin represented Fremantle, WA, from 1928 to 1931 and was defeated.
  Fact 0.94: John Curtin represented Fremantle, WA, from 1934 to 1945 and died in office.
  Fact 0.95: John Curtin led the 1st Curtin ministry.
  Fact 0.96: John Curtin won the 1943 election and led the 2nd Curtin ministry.
  Fact 0.97: Frank Forde (1890 to 1983) was Prime Minister from 6 July 1945 to 13 July 1945.
  Fact 0.98: Frank Forde was a Labor member.
  Fact 0.99: Frank Forde represented Capricornia, Qld, from 1922 to 1946 and was defeated.
  Fact 0.100: Frank Forde led the Forde ministry.
  Fact 0.101: Ben Chifley (1885 to 1951) was Prime Minister from 13 July 1945 to 19 December 1949.
  Fact 0.102: Ben Chifley was a Labor member.
  Fact 0.103: Ben Chifley represented Macquarie, NSW, from 1928 to 1931 and was defeated.
  Fact 0.104: Ben Chifley represented Macquarie, NSW, from 1940 to 1951 and died in office.
  Fact 0.105: Ben Chifley led the 1st Chifley ministry.
  Fact 0.106: Ben Chifley won the 1946 election and led the 2nd Chifley ministry.
  Fact 0.107: Sir Robert Menzies (1894 to 1978) was Prime Minister from 19 December 1949 to 26 January 1966.
  Fact 0.108: Robert Menzies was a Liberal (Coalition) member in this term.
  Fact 0.109: Robert Menzies represented Kooyong, Vic, from 1934 to 1966 and resigned.
  Fact 0.110: Robert Menzies won elections in 1949, 1951, 1954, 1955, 1958, 1961 and 1963.
  Fact 0.111: Robert Menzies led the 4th through 10th Menzies ministries.
  Fact 0.112: Harold Holt (1908 to 1967) was Prime Minister from 26 January 1966 to 19 December 1967 and disappeared in office.
  Fact 0.113: Harold Holt was a Liberal (Coalition) member.
  Fact 0.114: Harold Holt represented Fawkner, Vic, from 1935 to 1949.
  Fact 0.115: Harold Holt represented Higgins, Vic, from 1949 to 1967, ending when he disappeared.
  Fact 0.116: Harold Holt led the 1st Holt ministry.
  Fact 0.117: Harold Holt won the 1966 election and led the 2nd Holt ministry.
  Fact 0.118: John McEwen (1900 to 1980) was Prime Minister from 19 December 1967 to 10 January 1968.
  Fact 0.119: John McEwen was a Country (Coalition) member.
  Fact 0.120: John McEwen represented Echuca, Vic, from 1934 to 1937.
  Fact 0.121: John McEwen represented Indi, Vic, from 1937 to 1949.
  Fact 0.122: John McEwen represented Murray, Vic, from 1949 to 1971 and resigned.
  Fact 0.123: John McEwen led the McEwen ministry.
  Fact 0.124: John Gorton (1911 to 2002) was Prime Minister from 10 January 1968 to 10 March 1971.
  Fact 0.125: John Gorton was a Liberal (Coalition) member.
  Fact 0.126: John Gorton was a Senator from 1950 to 1968 and resigned.
  Fact 0.127: John Gorton was MP for Higgins, Vic, from 1968 to 1975 and retired.
  Fact 0.128: John Gorton led the 1st Gorton ministry.
  Fact 0.129: John Gorton won the 1969 election and led the 2nd Gorton ministry.
  Fact 0.130: William McMahon (1908 to 1988) was Prime Minister from 10 March 1971 to 5 December 1972.
  Fact 0.131: William McMahon was a Liberal (Coalition) member.
  Fact 0.132: William McMahon represented Lowe, NSW, from 1949 to 1982 and resigned.
  Fact 0.133: William McMahon led the McMahon ministry.
  Fact 0.134: Gough Whitlam (1916 to 2014) was Prime Minister from 5 December 1972 to 11 November.
  Fact 0.135: Gough Whitlam was a Labor member.
  Fact 0.136: Gough Whitlam represented Werriwa, NSW, from 1952 to 1978 and resigned.
  Fact 0.137: Gough Whitlam won the 1972 election and led the 1st Whitlam ministry.
  Fact 0.138: Gough Whitlam led the 2nd Whitlam ministry.
  Fact 0.139: Gough Whitlam led the 3rd Whitlam ministry.
  Fact 0.140: Malcolm Fraser (1930 to 2015) was Prime Minister from 11 November to 11 March.
  Fact 0.141: Malcolm Fraser was a Liberal (Coalition) member.
  Fact 0.142: Malcolm Fraser represented Wannon, Vic, from 1955 to 1983 and resigned.
  Fact 0.143: Malcolm Fraser led the 1st Fraser ministry.
  Fact 0.144: Malcolm Fraser led the 2nd Fraser ministry.
  Fact 0.145: Malcolm Fraser won the 1977 election and led the 3rd Fraser ministry.
  Fact 0.146: Malcolm Fraser won the 1980 election and led the 4th Fraser ministry.
  Fact 0.147: Bob Hawke (born 1929) was Prime Minister from 11 March to 20 December 1991.
  Fact 0.148: Bob Hawke was a Labor member.
  Fact 0.149: Bob Hawke represented Wills, Vic, from 1980 to 1992 and resigned.
  Fact 0.150: Bob Hawke led the 1st Hawke ministry.
  Fact 0.151: Bob Hawke won the 1984 election and led the 2nd Hawke ministry.
  Fact 0.152: Bob Hawke led the 3rd Hawke ministry.
  Fact 0.153: Bob Hawke won the 1990 election and led the 4th Hawke ministry.
  Fact 0.154: Paul Keating (born 1944) was Prime Minister from 20 December 1991 to 11 March.
  Fact 0.155: Paul Keating was a Labor member.
  Fact 0.156: Paul Keating represented Blaxland, NSW, from 1969 to 1996 and resigned.
  Fact 0.157: Paul Keating led the 1st Keating ministry.
  Fact 0.158: Paul Keating won the 1993 election and led the 2nd Keating ministry.
  Fact 0.159: John Howard (born 1939) was Prime Minister from 11 March to 3 December 2007.
  Fact 0.160: John Howard was a Liberal (Coalition) member.
  Fact 0.161: John Howard represented Bennelong, NSW, from 1974 to 2007 and was defeated.
  Fact 0.162: John Howard led the 1st Howard ministry.
  Fact 0.163: John Howard won the 1998 election and led the 2nd Howard ministry.
  Fact 0.164: John Howard won the 2001 election and led the 3rd Howard ministry.
  Fact 0.165: John Howard led the 4th Howard ministry.
  Fact 0.166: Kevin Rudd (born 1957) was Prime Minister from 3 December 2007 to 24 June.
  Fact 0.167: Kevin Rudd was a Labor member.
  Fact 0.168: Kevin Rudd represented Griffith, Qld, from 1998 to 2013 and resigned.
  Fact 0.169: Kevin Rudd won the 2007 election and led the 1st Rudd ministry.
  Fact 0.170: Julia Gillard (born 1961) was Prime Minister from 24 June to 27 June 2013.
  Fact 0.171: Julia Gillard was a Labor member.
  Fact 0.172: Julia Gillard represented Lalor, Vic, from 1998 to 2013 and retired.
  Fact 0.173: Julia Gillard led the 1st Gillard ministry.
  Fact 0.174: Julia Gillard led the 2nd Gillard ministry.
  Fact 0.175: Kevin Rudd was Prime Minister again from 27 June 2013 to 18 September 2013.
  Fact 0.176: Kevin Rudd was a Labor member.
  Fact 0.177: Kevin Rudd represented Griffith, Qld, from 1998 to 2013 and resigned.
  Fact 0.178: Kevin Rudd led the 2nd Rudd ministry.
  Fact 0.179: Tony Abbott (born 1957) was Prime Minister from 18 September 2013 to 15 September 2015.
  Fact 0.180: Tony Abbott was a Liberal (Coalition) member.
  Fact 0.181: Tony Abbott has represented Warringah, NSW, since 1994.
  Fact 0.182: Tony Abbott won the 2013 election and led the Abbott ministry.
  Fact 0.183: Malcolm Turnbull (born 1954) has been Prime Minister since 15 September 2015.
  Fact 0.184: Malcolm Turnbull is a Liberal (Coalition) member.
  Fact 0.185: Malcolm Turnbull has represented Wentworth, NSW, since 2004.
  Fact 0.186: Malcolm Turnbull led the 1st Turnbull ministry.
  Fact 0.187: Malcolm Turnbull won the 2016 election and led the 2nd Turnbull ministry.
  Fact 0.188: Malcolm Turnbull is still in office.
Sentence 1: Federation occurred on 1 January 1901 , but elections for the first parliament were not scheduled until late March . In the interim , an unelected caretaker government was necessary . In what is now known as the Hopetoun Blunder , the governor - general , Lord Hopetoun , invited Sir William Lyne , the premier of the most populous state , New South Wales , to form a government . Lyne was unable to do so and returned his commission in favour of Edmund Barton , who became the first Prime Minister and led the inaugural government into and beyond the election .
  Fact 1.0: Federation occurred on 1 January 1901.
  Fact 1.1: Elections for the first parliament were not scheduled until late March.
  Fact 1.2: In the interim, an unelected caretaker government was necessary.
  Fact 1.3: The event is now known as the Hopetoun Blunder.
  Fact 1.4: The governor-general, Lord Hopetoun, invited Sir William Lyne to form a government.
  Fact 1.5: Sir William Lyne was the premier of New South Wales.
  Fact 1.6: New South Wales was the most populous state.
  Fact 1.7: Sir William Lyne was unable to form a government.
  Fact 1.8: Sir William Lyne returned his commission in favour of Edmund Barton.
  Fact 1.9: Edmund Barton became the first Prime Minister.
  Fact 1.10: Edmund Barton led the inaugural government into the election.
  Fact 1.11: Edmund Barton led the inaugural government beyond the election.

Retrieved items:
<item 1>
Prime Minister of Australia
Fraser accepted these terms and immediately advised a double dissolution . An election was called for 13 December , which the Liberal Party won in its own right
( although the Liberals governed in a coalition with the Country Party ) . Powers and role ( edit ) The first Prime Minister of Australia , Edmund Barton (
sitting second from left ) , with his Cabinet , 1901 . Most of the Prime Minister 's powers derive from being head of Government . In practice , the Federal
Executive Council will act to ratify all decisions made by the cabinet and decisions of the cabinet will always require the support of the Prime Minister . The
powers of the governor - general to grant Royal Assent to legislation , to dissolve and prorogue parliament , to call elections and to make appointments are
exercised on the advice of the Prime Minister .
</item 1>

<item 2>
Prime Minister of Australia
The leaders of the three parties , Alfred Deakin , George Reid and Chris Watson each served as Prime Minister before losing a vote of confidence . As a result
of the Labor Party 's split over conscription , Billy Hughes and his supporters were expelled from the Labor Party in November 1916 . He subsequently continued
on as prime minister at the head of the new National Labor Party , which had only 14 members out of a total of 75 in the House of Representatives . The
Commonwealth Liberal Party -- despite still forming the Official Opposition -- provided confidence and supply until February 1917 , when the two parties agree
to merge and form the Nationalist Party . During the 1975 constitutional crisis , on 11 November 1975 , the governor - general , Sir John Kerr , dismissed the
Labor Party 's Gough Whitlam as Prime Minister . Despite Labor holding a majority in the House of Representatives , Kerr appointed the Leader of the Opposition
, Liberal leader Malcolm Fraser as caretaker Prime Minister , conditional on the passage of the Whitlam government 's Supply bills through the Senate and the
calling of an election for both houses of parliament . Fraser accepted these terms and immediately advised a double dissolution .
</item 2>

<item 3>
Prime Minister of Australia
The governor - general , Lord Casey , commissioned the Leader of the Country Party , John McEwen , to form a government until the Liberal Party elected a new
leader . McEwen was Prime Minister for 23 days , until the election of ( then Senator ) John Gorton . Excluding the brief transition periods during changes of
government or leadership elections , there have only been a handful of cases where someone other than the leader of the majority party in the House of
Representatives was Prime Minister : Federation occurred on 1 January 1901 , but elections for the first parliament were not scheduled until late March . In the
interim , an unelected caretaker government was necessary . In what is now known as the Hopetoun Blunder , the governor - general , Lord Hopetoun , invited Sir
William Lyne , the premier of the most populous state , New South Wales , to form a government . Lyne was unable to do so and returned his commission in favour
of Edmund Barton , who became the first Prime Minister and led the inaugural government into and beyond the election . During the second parliament , three
parties ( Free Trade , Protectionist and Labor ) had roughly equal representation in the House of Representatives .
</item 3>

<item 4>
Prime Minister of Australia
Other commentators argue that the Governor General acted improperly in 1975 as Whitlam still retained the confidence of the House of Representatives , and there
are no generally accepted conventions to guide the use of the Governor General 's reserve powers in this circumstance. . However , there is no constitutional
requirement that the Prime Minister sit in the House of Representatives , or even be a member of the federal parliament ( subject to a constitutionally
prescribed limit of three months ) , though by convention this is always the case . The only case where a member of the Senate was appointed Prime Minister was
John Gorton , who subsequently resigned his Senate position and was elected as a member of the House of Representatives ( Senator George Pearce was acting Prime
Minister for seven months in 1916 while Billy Hughes was overseas ; and Senator Mathias Cormann acted for a few days in February 2018 while Malcolm Turnbull was
overseas ) . Malcolm Turnbull has held the office of Prime Minister since 15 September 2015 . He received his commission after replacing Tony Abbott as the
leader of the Liberal Party , the dominant party in the Coalition government , following the outcome of the September 2015 Liberal leadership ballot .
</item 4>

<item 5>
Prime Minister of Australia
Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October
1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 †
Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945
Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW ,
</item 5>

<item 6>
Prime Minister of Australia
NSW , 1928 -- 1931 ( defeated ) ; 1940 -- 1951 ( died ) -- 1st Chifley 1946 2nd Chifley ( 12 ) Sir Robert Menzies ( 1894 -- 1978 ) Liberal ( Coalition ) 19
December 1949 26 January 1966 Kooyong , Vic , 1934 -- 1966 ( resigned ) 1949 4th Menzies 1951 5th Menzies 1954 6th Menzies 1955 7th Menzies 1958 8th Menzies
1961 9th Menzies 1963 10th Menzies 17 Harold Holt ( 1908 -- 1967 ) Liberal ( Coalition ) 26 January 1966 19 December 1967 † Fawkner , Vic , 1935 -- 1949 Higgins
, Vic ,
</item 6>

<item 7>
Prime Minister of Australia
and Administrators Premiers and Chief Ministers Parliaments and Assemblies State courts State electoral systems Local government ( show ) NSW Vic Qld WA SA Tas
NT Related topics ( show ) Regions Foreign relations Defence Force Public holidays Republicanism Other countries Atlas Australia 's first Prime Minister ,
Edmund Barton at the central table in the House of Representatives in 1901 . The Prime Minister of Australia is appointed by the Governor - General of Australia
under Section 64 of the Australian Constitution , which empowers the Governor - General , as the official representative of the monarch , to appoint government
ministers of state and requires them to be members of the House of Representatives or the Senate , or become members within three months of the appointment .
The Prime Minister and Treasurer are traditionally members of the House , but the Constitution does not have such a requirement . Before being sworn in as a
Minister of the Crown , a person must first be sworn in as a member of the Federal Executive Council if they are not already a member . Membership of the
Federal Executive Council entitles the member to the style of The Honourable ( usually abbreviated to The Hon ) for life , barring exceptional circumstances .
The senior members of the Executive Council constitute the Cabinet of Australia .
</item 7>

<item 8>
Prime Minister of Australia
The formal title of the portfolio has always been simply `` Prime Minister '' , except for the period of the Fourth Deakin Ministry ( June 1909 to April 1910 )
, when it was known as `` Prime Minister ( without portfolio ) '' . If a government can not get its appropriation ( budget ) legislation passed by the House of
Representatives , or the House passes a vote of `` no confidence '' in the government , the Prime Minister is bound by convention to immediately advise the
Governor - General to dissolve the House of Representatives and hold a fresh election . Following a resignation in other circumstances , or the death of a Prime
Minister , the governor - general will generally appoint as Prime Minister the person elected as leader by the governing party or , in the case of a coalition ,
the senior party in the coalition . There have been four notable exceptions to this : When Joseph Lyons , Prime Minister and leader of the United Australia
Party ( UAP ) , died suddenly in April 1939 , the governor - general , Lord Gowrie , called on Sir Earle Page to become caretaker Prime Minister . Page was the
leader of the smaller party in the governing coalition , the Country Party .
</item 8>

<item 9>
Prime Minister of Australia
NSW , 1969 -- 1996 ( resigned ) -- 1st Keating 1993 2nd Keating 25 John Howard ( 1939 -- ) Liberal ( Coalition ) 11 March 3 December 2007 Bennelong , NSW , 1974
-- 2007 ( defeated ) 1st Howard 1998 2nd Howard 2001 3rd Howard 4th Howard 26 Kevin Rudd ( 1957 -- ) Labor 3 December 2007 24 June Griffith , Qld , 1998 -- 2013
( resigned ) 2007 1st Rudd 27 Julia Gillard ( 1961 -- ) Labor 24 June 27 June 2013 Lalor , Vic , 1998 -- 2013 ( retired ) -- 1st Gillard 2nd Gillard ( 26 )
Kevin Rudd ( 1957 -- ) Labor 27 June 2013 18 September 2013 Griffith , Qld , 1998 -- 2013 ( resigned ) -- 2nd Rudd 28 Tony Abbott ( 1957 -- ) Liberal (
Coalition ) 18 September 2013 15 September 2015 Warringah ,
</item 9>

<item 10>
Prime Minister of Australia
He served under the United Australia Party and the Liberal Party respectively . The shortest - serving Prime Minister was Frank Forde , who was appointed to the
position on 6 July 1945 after the death of John Curtin , and served until 13 July 1945 when Ben Chifley was elected leader of the Australian Labor Party .
Parties Australian Labor Party Liberal Party of Australia Australian Country Party Nationalist Party of Australia United Australia Party Commonwealth Liberal
Party National Labor Party Free Trade Party Protectionist Party No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won
Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred
Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 )
Labour 27 April 1904 18 August 1904 Bland , NSW ,
</item 10>
=== END ITEM id=50c9bd1e1cc2da58 ===

=== ITEM id=c6a20d4b77f49bee ===
Gold sentences and facts:
Sentence 0: Taron Egerton as Johnny , a teenage gorilla who wants to sing , though his father would rather have him follow his criminal footsteps .
  Fact 0.0: Taron Egerton plays Johnny.
  Fact 0.1: Johnny is a teenage gorilla.
  Fact 0.2: Johnny wants to sing.
  Fact 0.3: Johnny's father would rather have him follow his criminal footsteps.

Retrieved items:
<item 1>
Sing (2016 American film)
Universal now has `` Max '' and `` Lunch '' set up at Illumination MacGuff . Jump up ^ Kilday , Gregg ( December 9 , 2016 ) . `` Illumination 's Chris
Meledandri Talks Success Secrets , Rumors He 'll Head DreamWorks '' . The Hollywood Reporter . Retrieved January 22 , 2017 . Jump up ^ Keslassy , Elsa ( 2
December 2016 ) . `` Oscar 's French Animation Connection Draws on Deep Talent Pool '' . Jump up ^ Grobar , Matt ( December 5 , 2016 ) . `` ' Sing ' Director
Garth Jennings On Working With `` Wizards , '' The Learning Curve Of His Feature Animation Debut `` . Deadline . Retrieved December 7 , 2016 . Jump up ^ Hammond
, Pete ( August 23 , 2016 ) . `` Universal & Illumination To Launch ' Sing ' With Live Jennifer Hudson And Tori Kelly Performances At Toronto '' . Deadline .
Retrieved September 11 , 2016 . Jump up ^ McClintock , Pamela ( January 15 , 2014 ) .
</item 1>

<item 2>
Sing (2016 American film)
It was directed and written by Garth Jennings , co-directed by Christophe Lourdelet and starring the voices of Matthew McConaughey , Reese Witherspoon , Seth
MacFarlane , Scarlett Johansson , John C. Reilly , Taron Egerton , and Tori Kelly . The film is about a group of anthropomorphic animals that enter a singing
competition , hosted by a koala hoping to save his theater . The film includes more than 60 songs from famous artists and also has an original song by Stevie
Wonder and Ariana Grande called `` Faith , '' which was nominated for a Golden Globe . It screened on the Toronto International Film Festival on September 11 ,
2016 , premiered at Microsoft Theater on December 3 , 2016 and was released in the United States on December 21 , 2016 , by Universal Pictures . The film
received generally positive reviews and grossed $634 million worldwide . Along with The Secret Life of Pets , it marked the first time that Illumination
Entertainment released two feature films in the same year . A sequel , entitled Sing 2 , has been announced for a release on December 25 , 2020 .
</item 2>

<item 3>
Sing (2016 American film)
Production ( edit ) In January 2014 , it was announced that Garth Jennings would write and direct an animated comedy film for Universal Pictures and
Illumination Entertainment , about `` courage , competition and carrying a tune , '' which was originally titled Lunch , and then retitled as Sing . On January
14 , 2015 , Matthew McConaughey was cast in the film 's lead voice role . Chris Meledandri and Janet Healy produced the film . On June 17 , 2015 , it was
confirmed that McConaughey 's character is named Buster and that John C. Reilly would voice Eddie , a sheep and Buster 's best friend . In November 2015 , it
was announced that Reese Witherspoon , Seth MacFarlane , Scarlett Johansson , Tori Kelly and Taron Egerton had joined the cast of the film . The film features
65 pop songs , the rights to which cost 15 percent of the film 's $75 million budget . The animation was created entirely in France by Illumination MacGuff .
</item 3>

<item 4>
Sing (2016 American film)
Sing ( 2016 American film ) - wikipedia Sing ( 2016 American film ) Jump to : navigation , search Sing Theatrical release poster Directed by Garth Jennings
Produced by Chris Meledandri Janet Healy Written by Garth Jennings Starring Matthew McConaughey Reese Witherspoon Seth MacFarlane Scarlett Johansson John C.
Reilly Taron Egerton Tori Kelly Music by Joby Talbot Edited by Gregory Perler Production company Illumination Entertainment Distributed by Universal Pictures
Release date September 11 , 2016 ( 2016 - 09 - 11 ) ( TIFF ) December 21 , 2016 ( 2016 - 12 - 21 ) ( United States ) Running time 108 minutes Country United
States Language English Budget $75 million Box office $634.1 million Sing is a 2016 American computer - animated musical comedy - drama film produced by
Illumination Entertainment .
</item 4>

<item 5>
Sing (2016 American film)
USA Today . Retrieved November 11 , 2015 . ^ Jump up to : Hopewell , John ( June 17 , 2015 ) . `` Annecy : Chris Meledandri on Matthew McConaughey as a Koala
Legit Theater Owner , ' The Grinch ' '' . Variety . Retrieved August 3 , 2015 . ^ Jump up to : Trumbore , Dave ( November 11 , 2015 ) . `` Meet the Animated
Animals of Illumination Entertainment 's 2016 Musical Comedy , ' Sing ' '' . Collider . Retrieved November 12 , 2015 . Jump up ^ Buxton , Adam ( April 14 , 2016
) . `` EP. 16 - Pub Walk With Garth Jennings '' . soundcloud.com . Retrieved April 17 , 2016 . Jump up ^ Yuen , Jenny ( September 11 , 2016 ) . `` TIFF 2016 :
Matthew McConaughey optimistic about ' Sing ' '' . Toronto Sun . Retrieved September 12 , 2016 .
</item 5>

<item 6>
Graph query row:
character: Johnny
relationship: RELATED_TO
role: None
relationship_detail: None
relation: None
voice_actor: Taron Egerton
</item 6>

<item 7>
Sing (2016 American film)
References ( edit ) ^ Jump up to : Felperin , Leslie ( September 11 , 2016 ) . `` ' Sing ' : Film Review '' . The Hollywood Reporter . Retrieved September 12 ,
2016 . ^ Jump up to : `` Sing ( 2016 ) '' . Box Office Mojo . Retrieved July 10 , 2017 . ^ Jump up to : Kit , Borys ( January 14 , 2015 ) . `` Matthew
McConaughey to Star in Illumination Animated Film ( Exclusive ) '' . The Hollywood Reporter . Retrieved August 3 , 2015 . Jump up ^ Amidi , Amid ( September 9 ,
2016 ) . `` A Complete List of Every Song in Illumination 's ' Sing ' '' . Cartoon Brew . Retrieved December 28 , 2016 . ^ Jump up to : Truitt , Brian (
November 10 , 2015 ) . `` Sneak peek : A-list animals ' SING ' together '' . USA Today . Retrieved November 11 , 2015 .
</item 7>

<item 8>
Sing (2016 American film)
Toronto Sun . Retrieved September 12 , 2016 . Jump up ^ `` Sing - Full Cast and Credits '' . Hollywood.com . Retrieved November 30 , 2016 . Jump up ^ Weintraub
, Steve ' Frosty ' ( 2016 - 12 - 19 ) . `` Garth Jennings on Getting Edgar Wright and Wes Anderson to Provide Voices in ' Sing ' '' . Collider . Retrieved 2017
- 11 - 09 . Jump up ^ `` ' Despicable Me 3 ′ Set For June 2017 Release ; ' Dr. Seuss ' How The Grinch Stole Christmas ' Arrives Five Months Later '' . Deadline
. January 15 , 2014 . Retrieved January 16 , 2014 . Jump up ^ Hopewell , John ( January 8 , 2014 ) . `` France Fights Bidding War to Lure Hollywood Filmmakers
'' . Variety . Retrieved January 9 , 2014 . Universal now has `` Max '' and `` Lunch '' set up at Illumination MacGuff .
</item 8>

<item 9>
Sing (2016 American film)
The animation was created entirely in France by Illumination MacGuff . Soundtrack ( edit ) Main article : Sing : Original Motion Picture Soundtrack A soundtrack
album for the film was released on December 21 , 2016 . Release ( edit ) The almost complete film was screened as a work in progress beginning September 11 ,
2016 at the Toronto International Film Festival . Universal Studios released the film on December 21 , 2016 . Home Media ( edit ) Sing was released on Blu - ray
, Blu - ray 3D , Ultra HD Blu - ray and DVD on March 21 , 2017 . It includes three short films : Gunter Babysits , Love at First Sight , and Eddie 's Life Coach
. In Gunter Babysits , Gunter offers to watch over Rosita and Norman 's piglets while they are away as he works to prove himself to be a good babysitter . In
Love at First Sight , Johnny sets Miss Crawly up with an online dating website where she manages to score a date with a lizard named Herman .
</item 9>

<item 10>
Sing (2016 American film)
Accolades ( edit ) List of awards and nominations Award Date of ceremony Recipient ( s ) Result Ref ( s ) AARP Annual Movies for Grownups Awards February 6 ,
2017 Best Movie for Grownups who Refuse to Grow Up Sing Nominated Annie Awards February 4 , 2017 Outstanding Achievement , Music in an Animated Feature
Production Joby Talbot Golden Globe Awards January 8 , 2017 Best Animated Feature Film Sing Best Original Song `` Faith '' -- Ryan Tedder , Stevie Wonder and
Francis Farewell Starlite Hollywood Music in Media Awards November 17 , 2016 Best Song -- Animated Film Best Soundtrack Album Sing : Original Motion Picture
Soundtrack Outstanding Music Supervision -- Film Jojo Villanueva Won Nickelodeon Kids ' Choice Awards March 11 , 2017 Favorite Animated Movie Sing Nominated
Favorite Voice From an Animated Movie Reese Witherspoon Most Wanted Pet Favorite Soundtrack Sing Saturn Awards June 28 , 2017 Best Animated Film Nominated
Sequel ( edit ) Universal and Illumination have announced plans for a sequel , tentatively titled Sing 2 , slated for December 25 , 2020 .
</item 10>
=== END ITEM id=c6a20d4b77f49bee ===

=== ITEM id=538ff59f70b2f150 ===
Gold sentences and facts:
Sentence 0: Adobe Flash Player Original author ( s ) FutureWave Macromedia Developer ( s ) Adobe Systems Initial release 1996 ; 22 years ago ( 1996 ) Stable release ( s ) ( ± ) Windows , macOS , Linux , Chrome OS 28.0. 0.137 / January 9 , 2018 ; 16 days ago ( 2018 - 01 - 09 ) Android 4.0. x 11.1. 115.81 / September 10 , 2013 ; 4 years ago ( 2013 - 09 - 10 ) Android 2. x and 3. x 11.1. 111.73 / September 10 , 2013 ; 4 years ago ( 2013 - 09 - 10 ) Solaris 11.2. 202.223 / March 28 , 2012 ; 5 years ago ( 2012 - 03 - 28 ) Preview release ( s ) ( ± ) Windows , macOS , Linux , Chrome OS 28.0. 0.152 Beta / January 24 , 2018 ; 1 day ago ( 2018 - 01 - 24 ) Written in C++ Operating system Windows , macOS , Linux , Chrome OS , Solaris , BlackBerry Tablet OS , Android , and Pocket PC Platform Web browsers and ActiveX - based software Available in Chinese Simplified , Chinese Traditional , English , French , German , Italian , Japanese , Polish , Russian , Spanish , Korean , and Turkish Type Run - time environment , Media player , and Browser extension License Freeware Website www.adobe.com/products/flashplayer.html
  Fact 0.0: Adobe Flash Player's original authors were FutureWave and Macromedia.
  Fact 0.1: Adobe Flash Player's developer is Adobe Systems.
  Fact 0.2: Adobe Flash Player's initial release was in 1996.
  Fact 0.3: Adobe Flash Player stable release 28.0.0.137 for Windows, macOS, Linux, and Chrome OS was released January 9, 2018.
  Fact 0.4: Adobe Flash Player stable release 11.1.115.81 for Android 4.0.x was released September 10, 2013.
  Fact 0.5: Adobe Flash Player stable release 11.1.111.73 for Android 2.x and 3.x was released September 10, 2013.
  Fact 0.6: Adobe Flash Player stable release 11.2.202.223 for Solaris was released March 28, 2012.
  Fact 0.7: Adobe Flash Player preview release 28.0.0.152 Beta for Windows, macOS, Linux, and Chrome OS was released January 24, 2018.
  Fact 0.8: Adobe Flash Player is written in C++.
  Fact 0.9: Adobe Flash Player runs on Windows, macOS, Linux, Chrome OS, Solaris, BlackBerry Tablet OS, Android, and Pocket PC.
  Fact 0.10: Adobe Flash Player's platform is web browsers and ActiveX-based software.
  Fact 0.11: Adobe Flash Player is available in Chinese Simplified, Chinese Traditional, English, French, German, Italian, Japanese, Polish, Russian, Spanish, Korean, and Turkish.
  Fact 0.12: Adobe Flash Player's type is run-time environment, media player, and browser extension.
  Fact 0.13: Adobe Flash Player is freeware.
  Fact 0.14: Adobe Flash Player's website is www.adobe.com/products/flashplayer.html.
Sentence 1: Availability on desktop operating systems Platform Latest version Browser support Windows XP and later Windows Server 2003 and later 28.0. 0.137 Firefox , Chrome , Chromium , Safari , Opera , Internet Explorer , Microsoft Edge Windows 2000 11.1. 102.55 ? Windows 98 and ME 9.0. 115.0 ? Windows 95 and NT 4 7.0. 14.0 ? Mac OS X 10.6 or later 28.0. 0.137 Firefox , Chrome , Chromium , Safari , Opera Mac OS X 10.5 10.3. 183.90 ? Classic Mac OS , PowerPC 7.0. 14.0 ? Classic Mac OS , 68k 5.0 ? Linux 28.0. 0.137 Firefox , Chrome , Chromium , Opera Solaris 11.2. 202.223 ?
  Fact 1.0: On Windows XP and later and Windows Server 2003 and later, the latest Adobe Flash Player version is 28.0.0.137.
  Fact 1.1: On Windows XP and later and Windows Server 2003 and later, Adobe Flash Player is supported in Firefox, Chrome, Chromium, Safari, Opera, Internet Explorer, and Microsoft Edge.
  Fact 1.2: On Windows 2000, the latest Adobe Flash Player version is 11.1.102.55, marked as uncertain in the source.
  Fact 1.3: On Windows 98 and ME, the latest Adobe Flash Player version is 9.0.115.0, marked as uncertain in the source.
  Fact 1.4: On Windows 95 and NT 4, the latest Adobe Flash Player version is 7.0.14.0, marked as uncertain in the source.
  Fact 1.5: On Mac OS X 10.6 or later, the latest Adobe Flash Player version is 28.0.0.137.
  Fact 1.6: On Mac OS X 10.6 or later, Adobe Flash Player is supported in Firefox, Chrome, Chromium, Safari, and Opera.
  Fact 1.7: On Mac OS X 10.5, the latest Adobe Flash Player version is 10.3.183.90, marked as uncertain in the source.
  Fact 1.8: On Classic Mac OS for PowerPC, the latest Adobe Flash Player version is 7.0.14.0, marked as uncertain in the source.
  Fact 1.9: On Classic Mac OS for 68k, the latest Adobe Flash Player version is 5.0, marked as uncertain in the source.
  Fact 1.10: On Linux, the latest Adobe Flash Player version is 28.0.0.137.
  Fact 1.11: On Linux, Adobe Flash Player is supported in Firefox, Chrome, Chromium, and Opera.
  Fact 1.12: On Solaris, the latest Adobe Flash Player version is 11.2.202.223, marked as uncertain in the source.

Retrieved items:
<item 1>
Adobe Flash Player
183.5 ) ( August 14 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.7 ) ( August 24 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.10 ) ( September
21 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.11 ) ( November 11 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.25 ) ( September 18 , 2012 )
Adobe Flash Player 10.3 ( version 10.3. 183.29 ) ( October 8 , 2012 ) Adobe Flash Player 11 ( version 11.0. 1.152 , codenamed Serrano ) ( October 4 , 2011 )
Desktop only Stage 3D accelerated graphics rendering Desktop : Windows ( DirectX 9 ) , OS X ( Intel processor only ) and Linux ( OpenGL 1.3 ) , SwiftShader
fallback Mobile : Android and iOS ( OpenGL ES 2 ) H.
</item 1>

<item 2>
Adobe Flash Player
0.134 , codenamed Octavia ) ( March 12 , 2015 ) Control Panel improvements Installer improvements for Mac Adobe Flash Player 18 ( version 18.0. 0.160 ,
codenamed ( Presidio of San Francisco Presidio ) ) ( June 9 , 2015 ) Contains fixes for Adobe Security Bulletin APSB 15 -- 11 Adobe Flash Player 19 ( version
19.0. 0.185 , codenamed Quint ) ( September 21 , 2015 ) Adobe Flash Player 20 ( version 20.0. 0.228 , codenamed Rankin ) ( December 8 , 2015 ) Adobe Flash
Player 21 ( version 21.0. 0.182 , codenamed Sutter ) ( March 10 , 2016 ) Adobe Flash Player 22 ( version 22.0. 0.192 , codenamed Townsend ) ( June 16 , 2016 )
Adobe Flash Player 23 ( version 23.0. 0.164 , codenamed Underwood ) ( September 13 , 2016 ) Adobe Flash Player 24 ( version 24.0.
</item 2>

<item 3>
Adobe Flash Player
0.11 , codenamed ( Atka ) ) ( November 16 , 2017 ) See also ( edit ) Adobe AIR Adobe Shockwave Apache Flex Microsoft Silverlight References and Notes ( edit ) ^
Jump up to : `` Adobe - Flash Player '' . Adobe.com . Adobe Systems . Retrieved 9 January 2018 . ^ Jump up to : `` Archived Flash Player versions '' . Adobe.com
. Adobe Systems . Retrieved 20 October 2015 . Jump up ^ `` Download Flash Player 28 Beta '' . labs.adobe.com . Adobe Systems . Retrieved 2018 - 01 - 25 . Jump
up ^ `` Adobe Flash Player Download '' . Adobe Systems . Retrieved August 10 , 2016 . Jump up ^ `` Why You Should Ditch Adobe Shockwave '' . Krebs on Security .
May 14 , 2014 . Retrieved February 21 , 2015 . Jump up ^ `` Adobe Flash Runtimes Statistics '' . Adobe Systems Incorporated . Retrieved January 2 , 2013 .
</item 3>

<item 4>
Adobe Flash Player
202.228 ) ( March 28 , 2012 ) Adobe Flash Player 11.2 ( version 11.2. 202.233 ) ( April 12 , 2012 ) Adobe Flash Player 11.2 ( version 11.2. 202.235 , codenamed
Brannan ) ( May 3 , 2012 ) The Windows version offers automatic updater options Dropped support of the browser plug - in for mobile devices ( Android ) .
Android app developers are encouraged to use Adobe Air and Android web developers should switch to HTML5 . Extended support for Flash player 11.2 on Solaris as
it is the last version to be supported . Adobe replaced Extended Support Release 11.2 on Linux with 24.0 on December 13 , 2016 . Adobe Flash Player 11.3 (
version 11.3. 300.257 ) ( June 8 , 2012 ) Adobe Flash Player 11.3 ( version 11.3. 300.262 ) ( June 21 , 2012 ) Adobe Flash Player 11.3 ( version 11.3.
</item 4>

<item 5>
Adobe Flash Player
0.186 , codenamed ( Van Ness Avenue Van Ness ) ) ( December 13 , 2016 ) Adobe Flash Player 24 ( version 24.0. 0.194 ) ( January 10 , 2017 ) Adobe Flash Player
25 ( version 25.0. 0.127 , codenamed Webster ) ( March 14 , 2017 ) Adobe Flash Player 25 ( version 25.0. 0.148 ) ( April 11 , 2017 ) Adobe Flash Player 25 (
version 25.0. 0.163 ) ( April 20 , 2017 ) Adobe Flash Player 26 ( version 26.0. 0.131 , codenamed York ) ( June 13 , 2017 ) Adobe Flash Player 27 ( version
27.0. 0.130 , codenamed Zoe ) ( September 12 , 2017 ) Adobe Flash Player 27 ( version 27.0. 0.159 ) ( October 10 , 2017 ) Adobe Flash Player 28 ( version 28.0.
</item 5>

<item 6>
Adobe Flash Player
0.137 Firefox , Chrome , Chromium , Safari , Opera , Internet Explorer , Microsoft Edge Windows 2000 11.1. 102.55 ? Windows 98 and ME 9.0. 115.0 ? Windows 95
and NT 4 7.0. 14.0 ? Mac OS X 10.6 or later 28.0. 0.137 Firefox , Chrome , Chromium , Safari , Opera Mac OS X 10.5 10.3. 183.90 ? Classic Mac OS , PowerPC 7.0.
14.0 ? Classic Mac OS , 68k 5.0 ? Linux 28.0. 0.137 Firefox , Chrome , Chromium , Opera Solaris 11.2. 202.223 ? Mobile platforms ( edit ) Main article : Adobe
Integrated Runtime In 2011 , Flash Player had emerged as the de facto standard for online video publishing on the desktop , with adaptive bitrate video
streaming , DRM , and fullscreen support .
</item 6>

<item 7>
Adobe Flash Player
Adobe Systems . November 11 , 2014 . Retrieved January 12 , 2015 . Jump up ^ `` 12 / 9 / 2014 -- Release -- Flash Player 16 '' . Adobe Systems . December 9 ,
2014 . Retrieved January 12 , 2015 . Jump up ^ `` 12 / 3 / 2015 -- Release -- Flash Player 17 '' . Adobe Systems . March 12 , 2014 . Retrieved March 13 , 2015 .
Jump up ^ `` Adobe Security Bulletin '' . Retrieved March 25 , 2017 . Further reading ( edit ) Understanding Flash Player with Adobe Scout -- an article
discussing the internals of the player and the Adobe Scout profiling tool External links ( edit ) Adobe Flash Player Flash Tester ( explains official old
working version check ) Adobe Flash File formats ActionScript ( AS ) Action Message Format ( AMF ) Flash Video ( FLV ) Flash XML Graphics ( FXG ) Local shared
objects ( LSOs ) MXML Flash Movie ( SWF ) Flash Code Library ( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark
</item 7>

<item 8>
Adobe Flash Player
adobe.com . Retrieved May 18 , 2012 . ^ Jump up to : `` Flash Player and Adobe AIR feature list '' . adobe.com . Retrieved May 9 , 2013 . Jump up ^ `` Flash
Player 12.0. 0.3 '' . Adobe Systems . November 14 , 2013 . Retrieved December 22 , 2013 . Jump up ^ `` Adobe Labs Downloads '' . Adobe . April 22 , 2015 .
Retrieved April 25 , 2015 . Jump up ^ `` Flash Player 13.0. 0.80 '' . Adobe Systems . January 28 , 2014 . Retrieved February 18 , 2014 . Jump up ^ `` 8 / 12 /
2014 -- Release -- Flash Player 14 '' . Adobe Systems . August 12 , 2014 . Retrieved January 12 , 2015 . Jump up ^ `` 11 / 11 / 2014 -- Release -- Flash Player
15 '' . Adobe Systems . November 11 , 2014 .
</item 8>

<item 9>
Adobe Flash Player
0.152 , codenamed ( Market Street ( San Francisco ) Market ) ) ( September 9 , 2014 ) Improved support for browser zoom levels Adobe Flash Player 15 ( version
15.0. 0.167 ) ( September 24 , 2014 ) Adobe Flash Player 15 ( version 15.0. 0.223 ) ( November 11 , 2014 ) Adobe Flash Player 15 ( version 15.0. 0.239 ) (
November 25 , 2014 ) Adobe Flash Player 16 ( version 16.0. 0.235 , codenamed Natoma ) ( December 12 , 2014 ) Stage3D -- Standard Constrained Profile PPAPI
Installers for Windows and Mac Adobe Flash Player 16 ( version 16.0. 0.257 ) ( January 13 , 2015 ) Adobe Flash Player 16 ( version 16.0. 0.287 ) ( January 22 ,
2015 ) Adobe Flash Player 17 ( version 17.0.
</item 9>

<item 10>
Adobe Flash Player
Adobe Flash Player - wikipedia Adobe Flash Player Jump to : navigation , search Adobe Flash Player Original author ( s ) FutureWave Macromedia Developer ( s )
Adobe Systems Initial release 1996 ; 22 years ago ( 1996 ) Stable release ( s ) ( ± ) Windows , macOS , Linux , Chrome OS 28.0. 0.137 / January 9 , 2018 ; 16
days ago ( 2018 - 01 - 09 ) Android 4.0. x 11.1. 115.81 / September 10 , 2013 ; 4 years ago ( 2013 - 09 - 10 ) Android 2. x and 3. x 11.1. 111.73 / September 10
, 2013 ; 4 years ago ( 2013 - 09 - 10 ) Solaris 11.2. 202.223 / March 28 , 2012 ; 5 years ago ( 2012 - 03 - 28 ) Preview release ( s ) ( ± ) Windows , macOS ,
Linux , Chrome OS 28.0.
</item 10>
=== END ITEM id=538ff59f70b2f150 ===
