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

=== ITEM id=c4ab9f4ff172373f ===
Gold sentences and facts:
Sentence 0: Lion 's mane jellyfish Scientific classification Kingdom : Animalia Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus , 1758 )
  Fact 0.0: The lion's mane jellyfish has the kingdom Animalia.
  Fact 0.1: The lion's mane jellyfish has the phylum Cnidaria.
  Fact 0.2: The lion's mane jellyfish has the class Scyphozoa.
  Fact 0.3: The lion's mane jellyfish has the order Semaeostomeae.
  Fact 0.4: The lion's mane jellyfish has the family Cyaneidae.
  Fact 0.5: The lion's mane jellyfish has the genus Cyanea.
  Fact 0.6: The lion's mane jellyfish has the species C. capillata.
  Fact 0.7: The binomial name of the lion's mane jellyfish is Cyanea capillata.
  Fact 0.8: The binomial name Cyanea capillata was given by Linnaeus in 1758.
Sentence 1: The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea , North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand . The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles 37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United States .
  Fact 1.0: The lion's mane jellyfish (Cyanea capillata) is also known as the giant jellyfish or the hair jelly.
  Fact 1.1: The lion's mane jellyfish is the largest known species of jellyfish.
  Fact 1.2: The range of the lion's mane jellyfish is confined to cold, boreal waters of the Arctic, northern Atlantic, and northern Pacific Oceans.
  Fact 1.3: The lion's mane jellyfish is common in the English Channel, Irish Sea, and North Sea.
  Fact 1.4: The lion's mane jellyfish is common in western Scandinavian waters south to Kattegat and Oresund.
  Fact 1.5: The lion's mane jellyfish may drift into the south-western part of the Baltic Sea.
  Fact 1.6: The lion's mane jellyfish cannot breed in the Baltic Sea due to the low salinity.
  Fact 1.7: Similar jellyfish, which may be the same species, are known to inhabit seas near Australia and New Zealand.
  Fact 1.8: The largest recorded specimen washed up on the shore of Massachusetts Bay in 1870.
  Fact 1.9: The largest recorded specimen had a bell with a diameter of 2.3 metres (7 ft 6 in).
  Fact 1.10: The largest recorded specimen had tentacles 37.0 m (121.4 ft) long.
  Fact 1.11: The lion's mane jellyfish has been observed below 42 degrees N latitude for some time.
  Fact 1.12: The lion's mane jellyfish has been observed in the larger bays of the east coast of the United States.

Retrieved items:
<item 1>
Lion's mane jellyfish
The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles
37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United
States . The lion 's mane jellyfish uses its stinging tentacles to capture , pull in and eat prey such as fish , sea creatures and smaller jellyfish . Contents
( hide ) 1 Taxonomy 2 Description 3 Ecology 4 Behavior and reproduction 5 Sting and human contact 6 In popular culture 7 Gallery 8 References 8.1 Other sources
9 External links Taxonomy ( edit ) Cyanea sp . The taxonomy of the Cyanea species is not fully agreed upon ; some zoologists have suggested that all species
within the genus should be treated as one .
</item 1>

<item 2>
Lion's mane jellyfish
wikipedia.org/w/index.php?title=Lion%27s_mane_jellyfish&oldid=834313563 '' Categories : Semaeostomeae Animals described in 1758 Hidden categories : Articles
with ' species ' microformats All articles with unsourced statements Articles with unsourced statements from February 2012 Talk Contents About Wikipedia
Wikispecies Azərbaycanca Català Cebuano Čeština Dansk Deutsch Español Esperanto فارسی Français Frysk Gaeilge 한국어 Bahasa Indonesia Íslenska Italiano עברית
Lietuvių Magyar Nederlands Norsk Norsk nynorsk پنجابی Polski Português Română Русский Simple English Suomi Svenska Türkçe Tiếng Việt Winaray 中文 26 more Edit
links This page was last edited on 5 April 2018 , at 01 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms
may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
</item 2>

<item 3>
Lion's mane jellyfish
Two distinct taxa , however , occur together in at least the eastern North Atlantic , with the blue jellyfish ( Cyanea lamarckii Péron & Lesueur , 1810 )
differing in color ( blue , not red ) and smaller size ( 10 -- 20 cm diameter , rarely 35 cm ) . Populations in the western Pacific around Japan are sometimes
distinguished as Cyanea nozakii , or as a subspecies , C. c. nozakii . In 2015 , Russian researchers announced a possible sister species , Cyanea tzetlinii
found in the White Sea , but this has not yet been recognized by other authoritative databases such as WoRMS or ITIS . Description ( edit ) Lion 's mane
jellyfish ( Cyanea capillata ) are named for their showy , trailing tentacles reminiscent of a lion 's mane . They can vary greatly in size : although capable
of attaining a bell diameter of over 2 metres ( 6 ft 7 in ) , those found in lower latitudes are much smaller than their far northern counterparts , with a bell
about 50 centimetres ( 20 in ) in diameter .
</item 3>

<item 4>
Lion's mane jellyfish
Lion 's mane jellyfish - wikipedia Lion 's mane jellyfish Jump to : navigation , search Lion 's mane jellyfish Scientific classification Kingdom : Animalia
Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus ,
1758 ) The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its
range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea ,
North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can
not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand .
</item 4>

<item 5>
Lion's mane jellyfish
Polar Biol . doi : 10.1007 / s00300 - 015 - 1707 - y . Jump up ^ `` Lion 's Mane Jellyfish '' . Scubatravel.co.uk . Retrieved 2010 - 09 - 03 . Jump up ^
https://web.archive.org/web/20110108024017/http://www.typesofjellyfish.net/lions_mane_jellyfish/lions_mane_jellyfish.html Jump up ^ Kozloff , Eugene ( 2003 ) .
Seashore Life of the Northern Pacific Coast . University of Washington Press . pp. 54 , 56 ( 1983 edition ) . ISBN 0295960841 . Jump up ^ Point 6 at ^ Jump up
to : `` Lions Mane Jellyfish '' . Types Of Jellyfish . Archived from the original on 2011 - 01 - 08 . Retrieved 2010 - 09 - 03 .
</item 5>

<item 6>
Lion's mane jellyfish
Size seems to dictate coloration : larger specimens are a vivid crimson to dark purple while smaller specimens grade to a lighter orange or tan , occasionally
colorless . The bell of the lion 's mane jellyfish is scalloped into eight lobes ( lappets ) , each lobe containing from 70 to 150 tentacles , arranged in four
fairly distinct rows . Along the bell margin is a balance organ at each of the eight indentations between the lobes -- the rhopalium -- which helps the
jellyfish orient itself . From the central mouth extend broad frilly oral arms with many stinging cells . Closer to its mouth , its total number of tentacles
are around 1200 . The long , thin tentacles which emanate from the bell 's subumbrella have been characterised as `` extremely sticky '' ; they also have
stinging cells . The tentacles of larger specimens may trail as long as 30 m ( 100 ft ) or more , with the tentacles of the longest known specimen measured at
37 m ( 120 ft ) in length .
</item 6>

<item 7>
Lion's mane jellyfish
This was later corrected -- in 1864 , a bootlace worm ( Lineus longissimus ) was found washed up on a Scottish shore that was 55 m ( 180 ft ) long . ( That
claim has its critics : because bootlace worms can easily stretch to several times their natural length , it is possible the worm did not actually grow to be
that length . ) Gallery ( edit ) A lion 's mane jelly capturing a sea gooseberry Lion 's mane jellyfish with full threads visible Lion 's mane jellyfish
contracted Lion 's mane jellyfish in star formation Lion 's mane jellyfish open , showing underside Life - sized model in the Smithsonian Museum of Natural
History , Washington DC , US Small , dead lion 's mane jelly washing up on the beach Blue lion 's mane jellyfish in Askeröfjorden outside Stenungsund , Sweden .
References ( edit ) Jump up ^ Hair Jelly ( Cyanea capillata ) . Australian Venom Research Unit . ^ Jump up to : `` Lion 's Mane Jellyfish -- Reference Library
'' . redOrbit . Retrieved 2010 - 09 - 03 .
</item 7>

<item 8>
Lion's mane jellyfish
redOrbit . Retrieved 2010 - 09 - 03 . Jump up ^ `` Lion 's Mane Jellyfish : Discovery Channel '' . Dsc.discovery.com. 2013 - 05 - 23 . Retrieved 2013 - 09 - 08
. Jump up ^ Glafira D. Kolbasova ; Arthur O. Zalevsky ; Azamat R. Gafurov ; Philipp O. Gusev ; Margarita A. Ezhova ; Anna A. Zheludkevich ; Olga P. Konovalova ;
Ksenia N. Kosobokova ; Nikita U. Kotlov ; Natalia O. Lanina ; Anna S. Lapashina ; Dmitry O. Medvedev ; Katerina S. Nosikova ; Ekaterina O. Nuzhdina ; Georgii A.
Bazykin ; Tatyana V. Neretina ( 2015 ) . `` A new species of Cyanea jellyfish sympatric to C. capillata in the White Sea '' . Polar Biol .
</item 8>

<item 9>
Lion's mane jellyfish
Considering the size of the species , it is possible that this mass incident was caused by a single specimen . In popular culture ( edit ) The lion 's mane
jellyfish appears in the Sherlock Holmes short story `` The Adventure of the Lion 's Mane '' published in The Case - Book of Sherlock Holmes . Holmes discovers
at the end of the story that the true killer of a professor who died shortly after going swimming ( shouting `` the lion 's mane '' before he succumbed ) was
actually this jellyfish . Suspicion was originally laid upon the professor 's rival in love , until the latter was similarly attacked ( he survived , although
badly stung ) . In the context of the story , it is only because the school professor has a weak heart that he succumbs , as is confirmed by the survival of the
second victim . A photograph widely distributed on the internet appears to show an anomalously large lion 's mane dwarfing a nearby diver by several times . The
photo was subsequently shown to be a hoax . On the popular television program QI the show claimed that the longest animal in the world was the lion 's mane
jellyfish .
</item 9>

<item 10>
Lion's mane jellyfish
Behavior and reproduction ( edit ) Main article : Jellyfish § Life cycle Lion 's mane jellyfish swimming , side view Lion 's mane jellyfish remain mostly very
near the surface , at no more than 20 m depth . Their slow pulsations drive them forwards weakly , so they depend on ocean currents to travel great distances .
The jellyfish are most often spotted during the late summer and autumn , when they have grown to a large size and the currents begin to sweep them to shore .
Like other jellyfish , lion 's manes are capable of both sexual reproduction in the medusa stage and asexual reproduction in the polyp stage . Lion 's mane
jellyfish have four different stages in their year - long lifespan : a larval stage , a polyp stage , an ephyrae stage , and the medusa stage . The female
jellyfish carries its fertilized eggs in its tentacle , where the eggs grow into larva . When the larva are old enough , the female deposits them on a hard
surface , where the larva soon grow into polyps . The polyps begin to reproduce asexually , creating stacks of small creatures called ephyrae .
</item 10>
=== END ITEM id=c4ab9f4ff172373f ===

=== ITEM id=ad1fb2803188d62c ===
Gold sentences and facts:
Sentence 0: Sinéad ( / ʃɪˈneɪd / shi - NADE ; Irish pronunciation : ( ˈʃɪnjeːd̪ɣ ) or ( ʃɪˈnjeːd̪ɣ ) ) , is an Irish feminine name . It is derived from the French Jeanette , which is cognate to the English Janet , itself a feminine form of the Hebrew Yohannan , `` God forgave / God gratified '' . In English , Sinéad is also commonly spelled Sinead . The name is generally translated into English as either Jane or Jennifer , or as the female version of the Scottish name Jean .
  Fact 0.0: Sinéad is an Irish feminine name.
  Fact 0.1: Sinéad is pronounced /ʃɪˈneɪd/, shi-NADE, in English.
  Fact 0.2: Sinéad has an Irish pronunciation of /ˈʃɪnjeːd̪ɣ/ or /ʃɪˈnjeːd̪ɣ/.
  Fact 0.3: Sinéad is derived from the French name Jeanette.
  Fact 0.4: Jeanette is cognate to the English name Janet.
  Fact 0.5: Janet is itself a feminine form of the Hebrew name Yohannan.
  Fact 0.6: Yohannan means 'God forgave' or 'God gratified'.
  Fact 0.7: Sinéad is also commonly spelled Sinead in English.
  Fact 0.8: The name Sinéad is generally translated into English as Jane or Jennifer.
  Fact 0.9: The name Sinéad is generally translated into English as the female version of the Scottish name Jean.

Retrieved items:
<item 1>
Sing (2016 American film)
The animation was created entirely in France by Illumination MacGuff . Soundtrack ( edit ) Main article : Sing : Original Motion Picture Soundtrack A soundtrack
album for the film was released on December 21 , 2016 . Release ( edit ) The almost complete film was screened as a work in progress beginning September 11 ,
2016 at the Toronto International Film Festival . Universal Studios released the film on December 21 , 2016 . Home Media ( edit ) Sing was released on Blu - ray
, Blu - ray 3D , Ultra HD Blu - ray and DVD on March 21 , 2017 . It includes three short films : Gunter Babysits , Love at First Sight , and Eddie 's Life Coach
. In Gunter Babysits , Gunter offers to watch over Rosita and Norman 's piglets while they are away as he works to prove himself to be a good babysitter . In
Love at First Sight , Johnny sets Miss Crawly up with an online dating website where she manages to score a date with a lizard named Herman .
</item 1>

<item 2>
Sinéad
Liam Maghnus Raghnall Risteárd Roibeárd Séarlas Somhairle Tiobóid Tormond Ualtar Ùisdean Bible - derived masculine names Abracham Adhamh Árón Biniáimin Daibhí
Dainéal Eafráim Gaibrial Íosac Maitiú Maois Mícheál ( Maidhc ) Nóe Seán Séamus Seosamh Síomón Solamh Tomás Bible - derived feminine names Abigeál Ánna Eábha
Eilís Léá Madailéin Máire Mallaidh Náoimí Ráichéal Ríobhca Rút Sinéad Siobhán Seacailín Seosaimhín Latin / Greek - derived masculine names Aindréas Antóin
Críostóir Damhnaic Pádraig Proinsias Giolla Críost Gréagóir Labhrás León Máirtín Nioclás Nollaig Peadar Pilib Pól Seoirse Stíofán Téodóir Uinseann See also
Goidelic languages
</item 2>

<item 3>
Sinéad
a character on the TV series Hollyoaks , played by Stephanie Davis Sinead Starling is a fictional character from The 39 Clues . Sinéad Ní Shúilleabháin , a
fictional character in The Wind That Shakes the Barley See also ( edit ) Siobhán , feminine Irish - language name ultimately derived from the same Hebrew name
References ( edit ) ( hide ) Gaelic given names Native masculine names Aodh Aodhán Aonghus Art Breandán Breasal Brian Cailean Cainnech Caoimhín Cathal Cearbhall
Cian Cianán Ciarán Cinnéididh Colm Conall Conán Conchobhar Cormac Criofan Dáire Déaglán Diarmaid Domhnall Donn Donnchadh Eachann Éibhear Eochaidh Eóghan
Fearghal Fearghus Féilim Fiach Fínghin Fionn Fionnbharr Fionnlagh Fionntán Flann Garaidh Iarlaith Lorcan Lughaidh Muircheartach Muiredach Muirgheas Murchadh
</item 3>

<item 4>
I before E except after C
Less strict interpretations admit as exceptions those words where eir , not preceded by c , represents the vowel of NEAR ( / ɪər / ) . Some categories of
exception : Many proper names , often because they are adopted from other languages . Fowler says the rule `` is useless with proper names '' ; Carney says ``
As one might expect of any rule , there are likely to be even more exceptions in names , many of which are Scottish '' : forenames and surnames Keith , Neil ,
Sheila , Stein , etc . placenames Leith , Keighley , Rheims , Raleigh , etc . Eid in the names of Muslim holidays ( Eid ul - Fitr , Eid al - Adha , etc . )
others like Cassiopeia Chemical names ending in - ein or - eine ( caffeine , casein , codeine , phthalein , protein , etc . ) . Here - ein ( e ) was originally
pronounced as two syllables / iː. ɪn / Scottish English words ( deil , deid , weill , etc . )
</item 4>

<item 5>
Sinéad
Sinéad - wikipedia Sinéad Jump to : navigation , search Sinéad ( / ʃɪˈneɪd / shi - NADE ; Irish pronunciation : ( ˈʃɪnjeːd̪ɣ ) or ( ʃɪˈnjeːd̪ɣ ) ) , is an Irish
feminine name . It is derived from the French Jeanette , which is cognate to the English Janet , itself a feminine form of the Hebrew Yohannan , `` God forgave
/ God gratified '' . In English , Sinéad is also commonly spelled Sinead . The name is generally translated into English as either Jane or Jennifer , or as the
female version of the Scottish name Jean . Sinéad may refer to : Contents ( hide ) 1 People 2 Fiction 3 See also 4 References People ( edit ) Sports Sinéad
Cahalan , camogie player Sinead Jennings , rower Sinead Kerr , ice dancer Sinéad Millea , former camogie player Sinead Miller , cyclist Sinead Russell , Olympic
Swimmer Music `` Sinéad '' ,
</item 5>

<item 6>
Sinéad
cyclist Sinead Russell , Olympic Swimmer Music `` Sinéad '' , a song by Within Temptation from their album The Unforgiving , and the name of a character in the
album 's tie - in comic book Sinead Harnett , singer / songwriter Sinéad Lohan , singer / songwriter Sinéad Madden , singer / songwriter Sinéad Mulvey , singer
, air hostess Sinéad O'Carroll , singer with Irish pop band B * Witched Sinéad O'Connor , singer / songwriter Sinéad Quinn , recording artist , reality show
contestant Film and television Sinéad Cusack , actress Sinead Desmond , TV presenter Sinead Keenan , actress Sinead Matthews , actress Sinéad Moynihan , model
and actress Sinéad Noonan , model and actress Other Sinéad de Valera , children 's book author Sinéad Morrissey , poet Sinéad Ní Loideáin , presenter Sinéad Ní
Neachtain , magazine editor Sinéad Sheppard , dancer Fiction ( edit ) Sinead O'Connor ( Hollyoaks ) , a character on the TV series Hollyoaks ,
</item 6>

<item 7>
I before E except after C
In the Great Vowel Shift , sounds ( eː ) and ( ɛː ) were raised to ( iː ) and ( eː ) respectively , with the latter subsequently becoming the diphthong ( eɪ ) .
Exceptionally , words in - ceive and their derivatives evolved from ( eː ) to ( iː ) rather than ( eɪ ) . History of the mnemonic ( edit ) The mnemonic ( in its
short form ) is found as early as 1866 , as a footnote in Manual of English Spelling , edited by schools inspector James Stuart Laurie from the work of a
Tavistock schoolmaster named Marshall . Michael Quinion surmises the rhyme was already established before this date . An 1834 manual states a similar rule in
prose ; others in 1855 and 1862 use different rhymes . Many textbooks from the 1870s on use the same rhyme as Laurie 's book .
</item 7>

<item 8>
Sinéad
Lughaidh Muircheartach Muiredach Muirgheas Murchadh Niall Oisín Rían Rónán Ruairí Searbhreathach Tadhg Toirdelbach Native feminine names Aíbinn Aifric Áine
Aisling Aoife Barrdhubh Bébinn Bríd Cacht Caoimhe Ciara Clíona Clodagh Cobhlaith Dearbhail Dearbhfhorghaill Deirbhile Deirdre Dubhchobhlaigh Eithne Éadaoin
Feidhelm Fíona Fionnuala Forbflaith Fíneamhain Gormflaith Gráinne Íde Lasairfhíona Meabh Mór Muirgheal Muirenn Niamh Nuala Órlaith Sadhbh Saoirse Síthmaith
Sláine Sorcha Úna Germanic - derived masculine names Amhlaoibh Annraoi Dubhghall Éamonn Gearóid Gofraid Ímar Lachlann Liam Maghnus Raghnall Risteárd Roibeárd
Séarlas
</item 8>

<item 9>
Graph query row:
entities: ['Sinéad', 'Jeanette', 'Janet', 'Yohannan']
relationships: [{'context': None, 'fact': None, 'relation': 'derived from', 'type': 'RELATED_TO'}, {'context': None, 'fact': None, 'relation': 'cognate to',
'type': 'RELATED_TO'}, {'context': None, 'fact': None, 'relation': 'feminine form of', 'type': 'RELATED_TO'}]
kind: name
date: None
</item 9>

<item 10>
Sinéad
Stíofán Téodóir Uinseann See also Goidelic languages Irish language Scottish Gaelic language Manx language Irish name List of Scottish Gaelic given names
Scottish Gaelic name Irish genealogy This page or section lists people that share the same given name . If an internal link led you here , you may wish to
change that link to point directly to the intended article . Retrieved from `` https://en.wikipedia.org/w/index.php?title= Sinéad&oldid = 792745813 ''
Categories : Given names Irish feminine given names Irish - language feminine given names Hidden categories : All set index articles Talk Contents About
Wikipedia Čeština Deutsch Español Français Italiano 日本 語 Edit links This page was last edited on 28 July 2017 , at 11 : 52 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=ad1fb2803188d62c ===

=== ITEM id=31b01e8dfac77ce6 ===
Gold sentences and facts:
Sentence 0: All I Want for Christmas Is You , a Night of Joy and Festivity is the first concert residency by American singer Mariah Carey . Performed annually at the Beacon Theatre in Manhattan , New York , the residency began on December 15 , 2014 , and will conclude on December 22 , 2017 , after completing six legs and thirty - four shows . Billboard revealed that the six dates had sold out with a total of 16,196 tickets while grossing $1.5 million during the first leg of the residency . The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You , with additional non-holiday tracks . The show also included `` a cast of forty , including a gospel choir , a clutch of carolers , several acrobats and a hyperactive round of kid singer / dancers . ''
  Fact 0.0: 'All I Want for Christmas Is You, a Night of Joy and Festivity' is the first concert residency by American singer Mariah Carey.
  Fact 0.1: The residency is performed annually at the Beacon Theatre in Manhattan, New York.
  Fact 0.2: The residency began on December 15, 2014.
  Fact 0.3: The residency will conclude on December 22, 2017.
  Fact 0.4: The residency completes six legs and thirty-four shows.
  Fact 0.5: Billboard revealed that the six dates had sold out with a total of 16,196 tickets.
  Fact 0.6: The six dates grossed $1.5 million during the first leg of the residency.
  Fact 0.7: The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You.
  Fact 0.8: The main set list for the show included additional non-holiday tracks.
  Fact 0.9: The show included a cast of forty.
  Fact 0.10: The show's cast included a gospel choir, a clutch of carolers, several acrobats, and a hyperactive round of kid singer/dancers.

Retrieved items:
<item 1>
All I Want for Christmas Is You, a Night of Joy and Festivity
of shows 6 in 2014 8 in 2015 9 in 2016 11 in 2017 34 in total Attendance 65,195 ( 93 % ) Box office $7,335,589 Mariah Carey concert chronology The Elusive
Chanteuse Show ( 2014 ) All I Want for Christmas Is You , a Night of Joy & Festivity ( 2014 - 17 ) Number 1 to Infinity ( 2015 -- 17 ) All I Want for Christmas
Is You , a Night of Joy and Festivity is the first concert residency by American singer Mariah Carey . Performed annually at the Beacon Theatre in Manhattan ,
New York , the residency began on December 15 , 2014 , and will conclude on December 22 , 2017 , after completing six legs and thirty - four shows . Billboard
revealed that the six dates had sold out with a total of 16,196 tickets while grossing $1.5 million during the first leg of the residency .
</item 1>

<item 2>
All I Want for Christmas Is You, a Night of Joy and Festivity
All I Want for Christmas is You , a Night of Joy and Festivity - wikipedia All I Want for Christmas is You , a Night of Joy and Festivity Jump to : navigation ,
search All I Want for Christmas Is You , a Night of Joy and Festivity Residency by Mariah Carey Location North America Europe Venue Various Associated albums
Merry Christmas & Merry Christmas II You Start date December 15 , 2014 ( 2014 - 12 - 15 ) End date December 22 , 2017 ( 2017 - 12 - 22 ) Legs 6 No.
</item 2>

<item 3>
All I Want for Christmas Is You, a Night of Joy and Festivity
Jump up ^ Gamboa , Glenn ( October 31 , 2014 ) . `` Mariah Carey sets holiday shows at Beacon Theatre '' . Newsday . Retrieved January 28 , 2015 . ^ Jump up to
: Farber , Jim ( December 15 , 2014 ) . `` Mariah Carey overdoes herself at wavering Beacon Theatre show : concert review '' . Daily News . New York . Retrieved
January 28 , 2015 . ^ Jump up to : Stern , Bradley ( August 18 , 2014 ) . `` Mariah Carey Announces ' The Elusive Chanteuse Show ' : See The First Tour Dates ''
. Idolator . Archived from the original on October 6 , 2017 . Retrieved October 6 , 2017 . ^ Jump up to : Brandle , Lars ( September 18 , 2014 ) . `` Mariah
Carey Announces ' Elusive Chanteuse ' Tour of Australia '' . Billboard . Archived from the original on October 6 , 2017 .
</item 3>

<item 4>
All I Want for Christmas Is You, a Night of Joy and Festivity
The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You , with additional non-holiday tracks . The show also included
`` a cast of forty , including a gospel choir , a clutch of carolers , several acrobats and a hyperactive round of kid singer / dancers . '' Contents ( hide ) 1
Background and development 2 Critical reception 3 Commercial performance 4 Set list 5 Residency dates 5.1 Cancelled shows 6 References Background and
development ( edit ) Mariah Carey released her fourteenth studio album , titled Me . I Am Mariah ... The Elusive Chanteuse , on May 27 , 2014 . It debuted at
number three on the Billboard 200 album chart with opening week sales of 58,000 copies in the United States , and number one on the Top R&B / Hip - Hop Albums
chart . Carey embarked on her eighth concert tour called The Elusive Chanteuse Show in October and November later that year , whereby she performed concerts
throughout East Asia , South East Asia and Australasia .
</item 4>

<item 5>
Graph query row:
entity: All I Want for Christmas Is You, a Night of Joy and Festivity
date: None
relationship: LOCATED_IN
related: Beacon Theatre
relatedDate: None
relatedKind: venue
</item 5>

<item 6>
All I Want for Christmas Is You, a Night of Joy and Festivity
Jump up ^ http://www.torontosun.com/2017/07/19/mariah-carey-brings-christmas-concert-series-to-las-vegas Jump up ^
http://windsorstar.com/news/local-news/mariah-carey-cancels-caesars-windsor-show Jump up ^
http://www.etonline.com/mariah-carey-cancels-first-shows-christmas-concert-due-respiratory-infection-91147 Jump up ^
http://www.billboard.com/articles/columns/pop/8038463/mariah-carey-cancels-shows-christmas-tour Jump up ^
http://metro.co.uk/2017/11/25/mariah-carey-cancels-more-christmas-tour-shows-for-a-few-extra-days-rest-7107422/ Jump up ^ http://www.eonline.
</item 6>

<item 7>
All I Want for Christmas Is You, a Night of Joy and Festivity
195 / 74,089 ( 93 % ) $7,335,589 Cancelled shows ( edit ) Date Venue City Country Reason 2017 November 17 , 2017 Caesars Windsor Windsor Canada `` Artist
illness '' ( Upper respiratory infection ) November 18 , 2017 Chicago Theatre Chicago United States November 20 , 2017 Sands Bethlehem Event Center Bethlehem
November 21 , 2017 Seneca Niagara Events Center Niagara Falls November 22 , 2017 Hard Rock Rocksino Northfield Park Cleveland November 24 , 2017 MGM National
Harbor Washington , D.C. November 25 , 2017 Borgata Hotel Casino & Spa Atlantic City November 28 , 2017 Beacon Theatre New York City December 1 , 2017
References ( edit ) Jump up ^ Lopez , Francisco ( October 31 , 2014 ) . `` Mariah Carey 's ' All I Want For Christmas Is You ' performance at the Beacon NYC ''
. AXS.com . Retrieved January 28 , 2015 . Jump up ^ Gamboa , Glenn ( October 31 , 2014 ) .
</item 7>

<item 8>
All I Want for Christmas Is You, a Night of Joy and Festivity
I Am Mariah ... The Elusive Chanteuse Live albums MTV Unplugged Compilation albums # 1 's Greatest Hits The Remixes The Ballads The Essential Mariah Carey # 1
to Infinity Soundtrack albums Glitter Video albums The First Vision MTV Unplugged + 3 Here Is Mariah Carey Fantasy : Mariah Carey at Madison Square Garden
Around the World # 1 's The Adventures of Mimi Television VH1 Divas American Idol A Christmas Melody Mariah 's World Concert tours Music Box Tour Daydream World
Tour Butterfly World Tour Rainbow World Tour Charmbracelet World Tour The Adventures of Mimi Angels Advocate Tour The Elusive Chanteuse Show The Sweet Sweet
Fantasy Tour All the Hits Tour The # 1 's Tour Concert residencies Live at the Pearl All I Want for Christmas Is You , a Night of Joy and Festivity # 1 to
Infinity Fragrances M by Mariah Carey Luscious Pink Forever Companies Monarc Entertainment Butterfly MC Records Related topics Tommy Mottola Nick Cannon Lionel
Cole James Packer VH1 Divas Book Portal Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
All I Want for Christmas Is You, a Night of Joy and Festivity
eonline.com/news/896163/mariah-carey-further-postpones-christmas-tour-over-health-issues Book : Merry Christmas II You ( hide ) Mariah Carey Albums Songs Awards
and nominations Singles Videography Performances Studio albums Mariah Carey Emotions Music Box Merry Christmas Daydream Butterfly Rainbow Charmbracelet The
Emancipation of Mimi E = MC2 Memoirs of an Imperfect Angel Merry Christmas II You Me . I Am Mariah ...
</item 9>

<item 10>
All I Want for Christmas Is You, a Night of Joy and Festivity
wikipedia.org/w/index.php?title=All_I_Want_for_Christmas_Is_You,_a_Night_of_Joy_and_Festivity&oldid=819618785 '' Categories : 2014 concert residencies 2015
concert residencies 2016 concert residencies 2017 concert residencies Mariah Carey concert residencies Christmas music Hidden categories : Pages using div col
with deprecated parameters Talk Contents About Wikipedia Español Bahasa Indonesia Português Edit links This page was last edited on 10 January 2018 , at 10 : 27
. Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of
Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=31b01e8dfac77ce6 ===

=== ITEM id=0f4bcc0810a05f28 ===
Gold sentences and facts:
Sentence 0: A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie , starring Jane Fonda and Lily Tomlin .
  Fact 0.0: A cover of a song by Stacey Markus is used at the end of episode nine, 'The Invitation', of Grace and Frankie.
  Fact 0.1: Grace and Frankie is a 2015 Netflix original series.
  Fact 0.2: Grace and Frankie stars Jane Fonda.
  Fact 0.3: Grace and Frankie stars Lily Tomlin.

Retrieved items:
<item 1>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 1>

<item 2>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 3>

<item 4>
Let My Love Open the Door
Chris Thomas Pete Townshend UK singles chronology `` Rough Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) `` Rough
Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) Pete Townshend US singles chronology `` Keep Me Turning '' ( 1977 ) (
with Ronnie Lane ) Keep Me Turning1977 `` Let My Love Open the Door '' ( 1980 ) Let My Love Open the Door1980 `` A Little Is Enough '' ( 1980 ) A Little Is
Enough 1980 `` Let My Love Open the Door '' is a song written and performed by Pete Townshend from his 1980 album Empty Glass . It reached the top ten in the
United States in that same year , reaching number nine . It reached number five in Canada .
</item 4>

<item 5>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 5>

<item 6>
Let My Love Open the Door
It reached number five in Canada . Contents ( hide ) 1 Background 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 Personnel 4 Covers 5 List of
Films , TV Shows , and Other Media featuring the track 6 References 7 External links Background ( edit ) Although Townshend is a devotee to the religious guru
Meher Baba , he claimed in the liner notes of his Anthology CD that `` Jesus sings '' on the track . `` Let My Love Open the Door '' was released as the second
single from Empty Glass in Britain , where it was backed with the non-album tracks `` Classified '' and `` Greyhound Girl . '' The song was a minor British hit
, reaching # 46 . The song saw more success when it was released as the debut single from Empty Glass in America , where the song reached # 9 . It was Pete
Townshend 's only solo top 10 hit on the Billboard Hot 100 , but The Who 's song `` I Can See For Miles , '' which was written by Townshend , reached the same
position on the chart 13 years earlier .
</item 6>

<item 7>
Let My Love Open the Door
The group Rogue Wave also covered it in `` Cover me '' released in 2017 . List of films , TV shows , and other media featuring the track ( edit ) The song has
been used frequently in film , most notably the comedy genre , often as trailer music for movies such as Jerry Maguire and How Do You Know , and then as end
credit song for Mr. Deeds in 2002 ; Jersey Girl and Along Came Polly , both in 2004 . It was featured in the following films and TV series : Look Who 's Talking
( 1989 ) Grosse Pointe Blank as `` E. Cola Mix '' ( 1997 ) Mr.
</item 7>

<item 8>
Let My Love Open the Door
Psychoderelict O Parvardigar Related articles Songs Songs Written Albums Peter Meaden The Who Deep End Tommy ( film ) The Who 's Tommy Tommy ( EP ) Tommy (
soundtrack ) The Boy Who Heard Music Quadrophenia ( musical ) Quadrophenia ( film ) Quadrophenia ( soundtrack ) The Lifehouse Method Horse 's Neck Who I Am Eel
Pie Publishing Eel Pie Studios Double O Cliff Townshend Emma Townshend Simon Townshend The Wick Ashdown House , Oxfordshire The Boathouse , Twickenham Chapel
House , Twickenham Retrieved from `` https://en.wikipedia.org/w/index.php?title=Let_My_Love_Open_the_Door&oldid=838422442 '' Categories : 1980 singles Debut
singles Pete Townshend songs Songs written by Pete Townshend 1980 songs Atco Records singles Song recordings produced by Pete Townshend British new wave songs
Hidden categories : Webarchive template wayback links Music infoboxes with deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia
Čeština Español Euskara Edit links This page was last edited on 26 April 2018 ,
</item 8>

<item 9>
Let My Love Open the Door
Initially , Townshend 's manager despised the track due to it `` not sounding like Townshend , '' and wanted it to be removed from Empty Glass . However , upon
the song 's chart success , his manager called to apologize . Despite the song 's critical and commercial success , Pete Townshend did not consider it one of
his best songs . He told Rolling Stone in an interview that `` Let My Love Open the Door '' was `` just a ditty , '' also claiming that he preferred his minor
US hit `` A Little Is Enough '' from the same album . In 1996 , Townshend released a new version of the song , called `` the E. Cola mix '' , turning the song
into a ballad . This version appeared in different television shows and film soundtracks . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 1980 ) Peak
position Canada RPM 100 5 UK Singles Chart 46 U.S. Billboard Hot 100 9 U.S. Cash Box Top 100 11 Year - end charts ( edit ) Chart ( 1980 ) Rank Canada 35 U.S.
</item 9>

<item 10>
Let My Love Open the Door
A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie ,
starring Jane Fonda and Lily Tomlin . The song is featured in a Walmart Christmas holiday television commercial which started airing in early December 2017 .
Lethal Weapon ( TV series ) , season 2 , episode 9 `` Fools Rush In '' play the song performed by `` Rogue wave '' ( 2017 ) . References ( edit ) Jump up ^
Unterberger , Richie ( 2012 - 06 - 14 ) . `` Let My Love Open the Door '' . Rolling Stone . Retrieved 2016 - 10 - 16 . Jump up ^ `` Pete Townshend , ' Empty
Glass ' 100 Best Albums of the Eighties '' . Rolling Stone . 1989 - 11 - 16 . Retrieved 2016 - 10 - 16 . ^ Jump up to : Bac-lac.gc.ca ^ Jump up to : `` Let My
Love Open the Door '' . Songfacts.com .
</item 10>
=== END ITEM id=0f4bcc0810a05f28 ===

=== ITEM id=5d322ec770361973 ===
Gold sentences and facts:
Sentence 0: Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20 nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond Canada ← 2014 2022 →
  Fact 0.0: The Ladies' singles event was held at the XXIII Olympic Winter Games.
  Fact 0.1: The Ladies' singles event was held at the Gangneung Ice Arena in Gangneung, South Korea.
  Fact 0.2: The Ladies' singles event was held on 21 to 23 February.
  Fact 0.3: The Ladies' singles event had 30 competitors from 20 nations.
  Fact 0.4: The winning score in the Ladies' singles event was 239.57.
  Fact 0.5: Alina Zagitova of Olympic Athletes from Russia was the first-place medalist.
  Fact 0.6: Evgenia Medvedeva of Olympic Athletes from Russia was the second-place medalist.
  Fact 0.7: Kaetlyn Osmond of Canada was the third-place medalist.
  Fact 0.8: The navigation links point to the previous event in 2014 and the next event in 2022.
Sentence 1: The ladies ' single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21 February , and the free skating was held on 23 February .
  Fact 1.0: The ladies' single figure skating competition was held at the 2018 Winter Olympics.
  Fact 1.1: The ladies' single figure skating competition was held at the Gangneung Ice Arena in Gangneung, South Korea.
  Fact 1.2: The short program was held on 21 February.
  Fact 1.3: The free skating was held on 23 February.

Retrieved items:
<item 1>
Figure skating at the 2018 Winter Olympics – Ladies' singles
97 20 23 Ivett Tóth Hungary 150.43 53.22 23 97.21 23 24 Isadora Williams Brazil 144.18 55.74 17 88.44 24 Did not advance to free skating 25 Larkyn Austman
Canada N / A 51.42 25 N / A N / A 26 Diāna Ņikitina Latvia N / A 51.12 26 N / A N / A 27 Giada Russo Italy N / A 50.88 27 N / A N / A 28 Anita Östlund Sweden N
/ A 49.14 28 N / A N / A 29 Anna Khnychenkova Ukraine N / A 47.59 29 N / A N / A 30 Aiza Mambekova Kazakhstan N / A 44.40 30 N / A N / A References ( edit )
Jump up ^ `` Venues '' . www.pyeongchang2018.com/ . Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics .
</item 1>

<item 2>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Contents ( hide ) 1 Records 2 Qualification 3 Schedule 4 Results 4.1 Short program 4.2 Free skating 4.3 Overall 5 References Records ( edit ) See also : World
and Olympic records set at the 2018 Winter Olympics For a complete list of figure skating records , see list of highest scores in figure skating . The following
new best scores were set during this competition : Component Skaters Score Date Ref Short program Evgenia Medvedeva ( OAR ) 81.61 21 February 2018 Alina
Zagitova ( OAR ) 82.92 The Olympic Athletes from Russia set two subsequent world records in the short program on February 21 , with Evgenia Medvedeva scoring
81.61 and then Alina Zagitova scoring 82.92 ten minutes later . Qualification ( edit ) Main article : Figure skating at the 2018 Winter Olympics --
Qualification A total of 30 skaters qualified to compete in the event , with each country allowed to only enter a maximum of three .
</item 2>

<item 3>
Figure skating at the 2018 Winter Olympics – Ladies' singles
TES - Technical Element Score ; PCS - Program Component Score SS - Skating Skills ; TR - Transitions ; PE - Performance / Execution CH - Choreography ; IN -
Interpretation ; Ded - Deduction ; StN - Starting Number Free skating ( edit ) The free skating was held on 23 February . Pl . Name Nation TSS TES PCS SS TR PE
CH IN Ded StN Evgenia Medvedeva Olympic Athletes from Russia * 156.65 * 79.18 * 77.47 * 9.57 9.43 9.82 9.71 9.89 0.00 24 Alina Zagitova Olympic Athletes from
Russia * 156.65 * 81.62 * 75.03 * 9.32 9.21 9.61 9.29 9.46 0.00 22 Kaetlyn Osmond Canada 152.15 76.50 75.65 9.50 9.21 9.57 9.39 9.61 0.00 23 Satoko Miyahara
Japan 146.
</item 3>

<item 4>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy Hamill ( USA ) 1980 : Anett Pötzsch ( GDR ) 1984 : Katarina Witt ( GDR ) 1988 : Katarina Witt ( GDR )
1992 : Kristi Yamaguchi ( USA ) 1994 : Oksana Baiul ( UKR ) 1998 : Tara Lipinski ( USA ) 2002 : Sarah Hughes ( USA ) 2006 : Shizuka Arakawa ( JPN ) 2010 : Yuna
Kim ( KOR ) 2014 : Adelina Sotnikova ( RUS ) 2018 : Alina Zagitova ( OAR ) Record scores in figure skating Combined total ( Total ) Short program ( SP ) Free
skating ( FS ) Short dance ( SD ) Free dance ( FD ) Juniors ( J ) Men Total : Yuzuru Hanyu 330.43 SP : Yuzuru Hanyu 112.72 FS : Yuzuru Hanyu 223.
</item 4>

<item 5>
Figure skating at the 2018 Winter Olympics – Ladies' singles
24 quotas were handed out during the 2017 World Figure Skating Championships and the remaining six were given out at the 2017 CS Nebelhorn Trophy . Each country
decided the entry of its teams , and athletes winning the quota were not necessarily granted the right to compete . All athletes competing must have met the
minimum total elements score , which does not include component scores . For the short program this was 20.00 and the free skate 36.00 . Schedule ( edit ) All
times are ( UTC + 9 ) . Date Time Round 21 February 10 : 00 Short program 23 February 10 : 00 Free skating Results ( edit ) Short program ( edit ) The short
program was held on 21 February . Pl . Name Nation TSS TES PCS SS TR PE CH IN Ded StN Alina Zagitova Olympic Athletes from Russia 82.92 WR 45.30 37.62 9.36 9.18
9.64 9.43 9.43 0.00 28 Evgenia Medvedeva Olympic Athletes from Russia 81.61 43.
</item 5>

<item 6>
Figure skating at the 2018 Winter Olympics – Ladies' singles
25 55.13 52.12 6.50 6.29 6.61 6.54 6.64 - 1.00 19 Gabrielle Daleman Canada 103.56 45.81 61.75 8.00 7.64 7.39 7.93 7.64 - 4.00 16 20 Li Xiangning China 101.97
50.56 51.41 6.57 6.21 6.46 6.39 6.50 0.00 5 21 Emmi Peltonen Finland 101.86 46.41 56.45 7.14 6.79 7.00 7.07 7.29 - 1.00 8 22 Alexia Paganini Switzerland 101.00
50.94 50.06 6.43 6.00 6.29 6.32 6.25 0.00 23 Ivett Tóth Hungary 97.21 49.82 50.39 6.
</item 6>

<item 7>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics . Retrieved 26 December 2017 . ^ Jump up to : `` Schedule '' . Retrieved 6 January
2018 . Jump up ^ Start list ^ Jump up to : Bieler , Des ; Janes , Chelsea ( 21 February 2018 ) . `` Alina Zagitova tops Russian teammate Evgenia Medvedeva 's
world - record short program score '' . The Washington Post . ISSN 0190 - 8286 . Retrieved 21 February 2018 . ^ Jump up to : `` Russia 's Zagitova beats
Medvedeva 's short program world record at Olympics '' . TASS. 21 February 2018 . Retrieved 21 February 2018 . Jump up ^ `` Qualification Systems for XXIII
Olympic Winter Games , PyeongChang 2018 , Figure skating '' . International Skating Union ( ISU ) . 17 May 2016 . Retrieved 20 January 2018 .
</item 7>

<item 8>
The women’s singles figure skating event at the 2018 PyeongChang Winter Olympics took place at Gangneung Ice Arena on February 21–23, with 30 skaters from 20
nations. Russia’s Alina Zagitova won gold with 239.57 points; teammate Evgenia Medvedeva took silver (238.26), and Canada’s Kaetlyn Osmond bronze (231.02).
Zagitova and Medvedeva set successive short-program world
</item 8>

<item 9>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Figure skating at the 2018 Winter Olympics -- ladies ' singles - wikipedia Figure skating at the 2018 Winter Olympics -- ladies ' singles Jump to : navigation ,
search Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20
nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond
Canada ← 2014 2022 → Figure skating at the 2018 Winter Olympics Qualification Singles men ladies Pairs mixed Ice dancing mixed Team event mixed The ladies '
single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21
February , and the free skating was held on 23 February .
</item 9>

<item 10>
Figure skating at the 2018 Winter Olympics – Ladies' singles
17 May 2016 . Retrieved 20 January 2018 . Jump up ^ `` Extract from ISU Communication 2097 '' . www.iceskating.org.uk/ . National Ice Skating Association ( UK )
. Retrieved 20 January 2018 . Jump up ^ Short program results Jump up ^ Free skating results Jump up ^ Final results Olympic champions in figure skating --
Ladies ' singles 1908 : Madge Syers ( GBR ) 1920 : Magda Julin ( SWE ) 1924 : Herma Szabo ( AUT ) 1928 : Sonja Henie ( NOR ) 1932 : Sonja Henie ( NOR ) 1936 :
Sonja Henie ( NOR ) 1948 : Barbara Ann Scott ( CAN ) 1952 : Jeannette Altwegg ( GBR ) 1956 : Tenley Albright ( USA ) 1960 : Carol Heiss ( USA ) 1964 : Sjoukje
Dijkstra ( NED ) 1968 : Peggy Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy
</item 10>
=== END ITEM id=5d322ec770361973 ===

=== ITEM id=e8eba5283304a8c0 ===
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
Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October
1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 †
Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945
Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW ,
</item 1>

<item 2>
Prime Minister of Australia
Prime Minister of Australia - Wikipedia Prime Minister of Australia This is the latest accepted revision , reviewed on 26 April 2018 . Jump to : navigation ,
search This article is about the office of the Prime Minister of Australia . For a list of Australian Prime Ministers , see List of Prime Ministers of Australia
. Prime Minister of Australia Commonwealth Coat of Arms Incumbent Malcolm Turnbull since 15 September 2015 Government of Australia Department of the Prime
Minister and Cabinet Style The Honourable ( formal ) Prime Minister ( spoken ) His / Her Excellency ( in international correspondence ) Member of Cabinet
National Security Committee Federal Executive Council Reports to Parliament , Governor - General Residence The Lodge Kirribilli House Seat Canberra Appointer
Governor - General of Australia by convention , based on appointee 's ability to command confidence in the House of Representatives Term length At the Governor
- General 's pleasure contingent on the Prime Minister 's ability to command confidence in the lower house of Parliament Inaugural holder Edmund Barton
Formation 1 January 1901 Deputy Michael McCormack Salary $527,852 ( AUD ) Website pm.gov.au The Prime Minister of Australia ( sometimes informally abbreviated
to PM ) is the head of government of Australia .
</item 2>

<item 3>
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
</item 3>

<item 4>
Prime Minister of Australia
The governor - general , Lord Casey , commissioned the Leader of the Country Party , John McEwen , to form a government until the Liberal Party elected a new
leader . McEwen was Prime Minister for 23 days , until the election of ( then Senator ) John Gorton . Excluding the brief transition periods during changes of
government or leadership elections , there have only been a handful of cases where someone other than the leader of the majority party in the House of
Representatives was Prime Minister : Federation occurred on 1 January 1901 , but elections for the first parliament were not scheduled until late March . In the
interim , an unelected caretaker government was necessary . In what is now known as the Hopetoun Blunder , the governor - general , Lord Hopetoun , invited Sir
William Lyne , the premier of the most populous state , New South Wales , to form a government . Lyne was unable to do so and returned his commission in favour
of Edmund Barton , who became the first Prime Minister and led the inaugural government into and beyond the election . During the second parliament , three
parties ( Free Trade , Protectionist and Labor ) had roughly equal representation in the House of Representatives .
</item 4>

<item 5>
Prime Minister of Australia
Fisher Cook Fisher Hughes Bruce Scullin Lyons Page Menzies Fadden Curtin Forde Chifley Menzies Holt McEwen Gorton McMahon Whitlam Fraser Hawke Keating Howard
Rudd Gillard Rudd Abbott Turnbull Lists of Prime Ministers of Australia By order Order of service Service ( graphical ) Statehood Historical rankings Military
service Time in office Age By professional life Political affiliation Spouses Prime Minister 's spouse Parliament of Australia Components Queen ( represented by
the Governor - General ) Senate House of Representatives Parliamentary officers Senate President of the Senate Clerk Leader of the Government in the Senate
Manager of Government Business Leader of the Opposition in the Senate Manager of Opposition Business Whips ( list ) House of Representatives Speaker of the
Australian House of Representatives Clerk Prime Minister Leader of the House Leader of the Opposition Manager of Opposition Business Whips ( list ) Members
Served in both houses Represented multiple states 30 + years Father of the Parliament Senate Current ( previous ) Appointees Women Father of the Senate House of
Representatives Current ( previous ) Women Father of the House Parliaments Chronology of Australian federal parliaments 44th 45th Parliament of Australia
Procedure Speech from the throne Joint meetings Question time Matter of public importance Readings of bills Queen 's Consent Royal
</item 5>

<item 6>
Prime Minister of Australia
Fraser accepted these terms and immediately advised a double dissolution . An election was called for 13 December , which the Liberal Party won in its own right
( although the Liberals governed in a coalition with the Country Party ) . Powers and role ( edit ) The first Prime Minister of Australia , Edmund Barton (
sitting second from left ) , with his Cabinet , 1901 . Most of the Prime Minister 's powers derive from being head of Government . In practice , the Federal
Executive Council will act to ratify all decisions made by the cabinet and decisions of the cabinet will always require the support of the Prime Minister . The
powers of the governor - general to grant Royal Assent to legislation , to dissolve and prorogue parliament , to call elections and to make appointments are
exercised on the advice of the Prime Minister .
</item 6>

<item 7>
Prime Minister of Australia
The leaders of the three parties , Alfred Deakin , George Reid and Chris Watson each served as Prime Minister before losing a vote of confidence . As a result
of the Labor Party 's split over conscription , Billy Hughes and his supporters were expelled from the Labor Party in November 1916 . He subsequently continued
on as prime minister at the head of the new National Labor Party , which had only 14 members out of a total of 75 in the House of Representatives . The
Commonwealth Liberal Party -- despite still forming the Official Opposition -- provided confidence and supply until February 1917 , when the two parties agree
to merge and form the Nationalist Party . During the 1975 constitutional crisis , on 11 November 1975 , the governor - general , Sir John Kerr , dismissed the
Labor Party 's Gough Whitlam as Prime Minister . Despite Labor holding a majority in the House of Representatives , Kerr appointed the Leader of the Opposition
, Liberal leader Malcolm Fraser as caretaker Prime Minister , conditional on the passage of the Whitlam government 's Supply bills through the Senate and the
calling of an election for both houses of parliament . Fraser accepted these terms and immediately advised a double dissolution .
</item 7>

<item 8>
Prime Minister of Australia
He served under the United Australia Party and the Liberal Party respectively . The shortest - serving Prime Minister was Frank Forde , who was appointed to the
position on 6 July 1945 after the death of John Curtin , and served until 13 July 1945 when Ben Chifley was elected leader of the Australian Labor Party .
Parties Australian Labor Party Liberal Party of Australia Australian Country Party Nationalist Party of Australia United Australia Party Commonwealth Liberal
Party National Labor Party Free Trade Party Protectionist Party No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won
Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred
Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 )
Labour 27 April 1904 18 August 1904 Bland , NSW ,
</item 8>

<item 9>
Prime Minister of Australia
The formal title of the portfolio has always been simply `` Prime Minister '' , except for the period of the Fourth Deakin Ministry ( June 1909 to April 1910 )
, when it was known as `` Prime Minister ( without portfolio ) '' . If a government can not get its appropriation ( budget ) legislation passed by the House of
Representatives , or the House passes a vote of `` no confidence '' in the government , the Prime Minister is bound by convention to immediately advise the
Governor - General to dissolve the House of Representatives and hold a fresh election . Following a resignation in other circumstances , or the death of a Prime
Minister , the governor - general will generally appoint as Prime Minister the person elected as leader by the governing party or , in the case of a coalition ,
the senior party in the coalition . There have been four notable exceptions to this : When Joseph Lyons , Prime Minister and leader of the United Australia
Party ( UAP ) , died suddenly in April 1939 , the governor - general , Lord Gowrie , called on Sir Earle Page to become caretaker Prime Minister . Page was the
leader of the smaller party in the governing coalition , the Country Party .
</item 9>

<item 10>
Prime Minister of Australia
Caretaker government Court of Disputed Returns Locations Parliament House , Melbourne Old Parliament House Parliament House , Canberra Politics of Australia
Commonwealth Monarch Governor - General Prime Minister Cabinet ( Shadow Cabinet ) Executive Council Ministry Foreign relations Parliament House of
Representatives Senate Opposition Leader High Court Lower courts Constitution Statute of Westminster Australia Act Federal elections pre-1969 1969 1972 1977
1980 1984 1990 1993 1998 2001 2007 2013 2016 next by - elections State / Territory governments Governors and Administrators Premiers and Chief Ministers
Parliaments and Assemblies NSW Vic Qld WA SA Tas ACT NT State / Territory elections Vic 2014 NSW 2015 ACT 2016 NT 2016 WA 2017 Qld 2017 Tas 2018 SA 2018 Vic
2018 NSW 2019 ACT 2020 NT 2020 WA 2021 Qld 2020 Tas 2022 SA 2022 Timeline of Elections Amalgamated Timeline Local government NSW Vic Qld WA SA Tas NT Political
parties Coalition ( Liberal , National , Liberal National , Country Liberal ) Conservatives Greens Hinch Katter Labor Liberal Democrat One Nation Xenophon Other
</item 10>
=== END ITEM id=e8eba5283304a8c0 ===
