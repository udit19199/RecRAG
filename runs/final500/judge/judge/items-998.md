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

=== ITEM id=04d4704b93fd3326 ===
Gold sentences and facts:
Sentence 0: `` Baby , Now That I 've Found You '' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh , Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' ( Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 ( From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by The Cox Family ) `` I Will '' ( John Lennon , Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album '' performed by Alison Krauss & Union Station )
  Fact 0.0: 'Baby, Now That I've Found You' was written by Tony Macaulay and John MacLeod.
  Fact 0.1: 'Baby, Now That I've Found You' runs 3 minutes 49 seconds.
  Fact 0.2: 'Baby, Now That I've Found You' is previously unreleased.
  Fact 0.3: 'Oh, Atlanta' was written by Mick Ralphs.
  Fact 0.4: 'Oh, Atlanta' runs 4 minutes 41 seconds.
  Fact 0.5: 'Oh, Atlanta' is previously unreleased.
  Fact 0.6: 'Broadway' was written by Sidney & Suzanne Cox.
  Fact 0.7: 'Broadway' runs 3 minutes 57 seconds.
  Fact 0.8: 'Broadway' is previously unreleased.
  Fact 0.9: 'Every Time You Say Goodbye' was written by John Pennell.
  Fact 0.10: 'Every Time You Say Goodbye' runs 3 minutes 13 seconds.
  Fact 0.11: 'Every Time You Say Goodbye' is from the album 'Every Time You Say Goodbye' by Alison Krauss & Union Station.
  Fact 0.12: 'Tonight I'll Be Lonely Too' was written by Sidney Cox.
  Fact 0.13: 'Tonight I'll Be Lonely Too' runs 3 minutes 25 seconds.
  Fact 0.14: 'Tonight I'll Be Lonely Too' is from the album 'I've Got That Old Feeling' by Alison Krauss.
  Fact 0.15: 'Teardrops Will Kiss the Morning Dew' was written by Paul Craft.
  Fact 0.16: 'Teardrops Will Kiss the Morning Dew' runs 3 minutes 38 seconds.
  Fact 0.17: 'Teardrops Will Kiss the Morning Dew' is from the album 'Two Highways' by Alison Krauss & Union Station.
  Fact 0.18: 'Sleep On' was written by Nelson Mandrell.
  Fact 0.19: 'Sleep On' runs 2 minutes 23 seconds.
  Fact 0.20: 'Sleep On' is from the album 'Too Late to Cry' by Alison Krauss.
  Fact 0.21: 'When God Dips His Pen of Love in My Heart' was written by Cleavant Derricks.
  Fact 0.22: 'When God Dips His Pen of Love in My Heart' runs 2 minutes 26 seconds.
  Fact 0.23: 'When God Dips His Pen of Love in My Heart' is from the album 'Everybody's Reaching Out for Someone' by The Cox Family.
  Fact 0.24: 'I Will' was written by John Lennon and Paul McCartney.
  Fact 0.25: 'I Will' runs 4 minutes 4 seconds.
  Fact 0.26: 'I Will' is from the album 'Within Reach' by Tony Furtado.
  Fact 0.27: 'I Don't Believe You've Met My Baby' was written by Autry Inman.
  Fact 0.28: 'I Don't Believe You've Met My Baby' runs 3 minutes 10 seconds.
  Fact 0.29: 'I Don't Believe You've Met My Baby' is from the album 'Slide Rule' by Jerry Douglas.
  Fact 0.30: 'In the Palm of Your Hand' was written by Ron Block.
  Fact 0.31: 'In the Palm of Your Hand' runs 3 minutes 24 seconds.
  Fact 0.32: 'In the Palm of Your Hand' is from the album 'I Know Who Holds Tomorrow' by Alison Krauss & The Cox Family.
  Fact 0.33: 'When You Say Nothing at All' was written by Paul Overstreet and Don Schlitz.
  Fact 0.34: 'When You Say Nothing at All' runs 4 minutes 21 seconds.
  Fact 0.35: 'When You Say Nothing at All' is performed by Alison Krauss & Union Station on the album 'Keith Whitley: A Tribute Album'.

Retrieved items:
<item 1>
What Ifs
The song is the first No. 1 on the chart for both Brown and Alaina . The following week , for charts dated October 28 , 2017 , `` What Ifs '' stayed at number
one , as well as topping the Country Airplay and Country Streaming Songs charts . Together with the release of the deluxe edition of the album which also
reached number on Top Country Albums , and another number one for a new track `` Heaven '' on digital sales , Brown became the first artist to have simultaneous
number ones on all five main country charts . The song was certified Platinum by the RIAA on October 2 , 2017 . It was the third most - consumed country song of
2017 , with 1,551,000 including streams ( 599,000 in pure sale ) sold in the year . It has sold 679,000 copies in the US as of March 2018 . Music video ( edit )
The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at
Newport Ranch , a resort and cattle ranch to the north of San Francisco .
</item 1>

<item 2>
Now That I've Found You: A Collection
It is a retrospective of the early part of Krauss ' recording career . It includes songs that appeared on her solo albums , albums by Alison Krauss & Union
Station , and some that appeared on an album by Alison Krauss & the Cox Family . It also contains some new material . The album features the hit country single
`` When You Say Nothing at All '' , sold over two million copies earning double platinum RIAA certification . The title single `` Baby , Now That I 've Found
You '' won the Grammy Award for Best Female Country Vocal Performance and `` When You Say Nothing at All '' was named Single of the Year by the Country Music
Association . Contents ( hide ) 1 Track listing 2 Personnel 3 Chart performance 3.1 Album 4 References Track listing ( edit ) `` Baby , Now That I 've Found You
'' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh ,
</item 2>

<item 3>
Now That I've Found You: A Collection
Brent Truitt : Mandolin Stuart Duncan : Fiddle , Harmonica , Cymbals Adam Steffey : Mandolin & Mandola Pete Wasner : Piano Chart performance ( edit ) Album (
edit ) Chart ( 1995 ) Peak position Canadian RPM Country Albums U.S. Billboard 200 13 U.S. Billboard Top Country Albums References ( edit ) Jump up ^ Allmusic
review Jump up ^ Robert Christgau review Jump up ^ Rolling Stone review ^ Jump up to : Alison Krauss Artist Chart Position at Billboard Now That I 've Found You
for Rounder Records . Retrieved August 28 , 2010 . Alison Krauss bio for CMT.com . Retrieved June 7 , 2006 . Now That I 've Found You on Amazon . Retrieved June
7 , 2006 . Preceded by The Woman in Me by Shania Twain RPM Country Albums number - one album October 30 - November 5 ,
</item 3>

<item 4>
Now That I've Found You: A Collection
Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the
Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison
Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album ''
performed by Alison Krauss & Union Station ) Personnel ( edit ) Alison Krauss : Vocal , Fiddle , Viola Ron Block : Vocal , Acoustic & Electric Guitars , Banjo
Russ Barenberg , Mike Marshall , Scott Nygaard , Tim Stafford : Guitars Jerry Douglas : Guitars , Dobro Sam Bush , Brent Truitt : Mandolin Stuart Duncan :
Fiddle , Harmonica ,
</item 4>

<item 5>
Now That I've Found You: A Collection
Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say
Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' (
Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 (
From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by
Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by
The Cox Family ) `` I Will '' ( John Lennon ,
</item 5>

<item 6>
Now That I've Found You: A Collection
1995 Succeeded by Starting Over by Reba McEntire ( hide ) Alison Krauss & Union Station Alison Krauss Barry Bales Ron Block Jerry Douglas Dan Tyminski John
Pennell Jeff White Brent Truitt Mike Harman Adam Steffey Tim Stafford Studio albums Different Strokes Too Late to Cry Two Highways I 've Got That Old Feeling
Every Time You Say Goodbye I Know Who Holds Tomorrow So Long So Wrong Forget About It New Favorite Lonely Runs Both Ways Paper Airplane Windy City Compilation
albums Now That I 've Found You : A Collection Live A Hundred Miles or More : A Collection Essential Alison Krauss With Robert Plant Raising Sand Singles ``
When You Say Nothing at All '' `` Baby Now That I 've Found You '' `` Baby Mine '' `` The Lucky One '' `` Missing You '' `` Please Read the Letter '' `` Rich
Woman '' Featured singles `` Teach Your Children '' `` Somewhere in the Vicinity of the Heart '' `` It 's Not Over '' `` Buy Me a Rose '' `` Whiskey Lullaby ''
Related articles Discography Awards and nominations Alison Brown Robert Plant Viktor Krauss Up !
</item 6>

<item 7>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Now That I've Found You: A Collection
Now That I 've Found You : a Collection - wikipedia Now That I 've Found You : a Collection Jump to : navigation , search Now That I 've Found You : A
Collection Compilation album by Alison Krauss Released February 7 , 1995 Genre Bluegrass / Country Length 42 : 31 Label Rounder Producer Alison Krauss & Union
Station Alison Krauss chronology I Know Who Holds Tomorrow ( 1994 ) I Know Who Holds Tomorrow 1994 Now That I 've Found You : A Collection ( 1995 ) So Long So
Wrong ( 1997 ) So Long So Wrong 1997 Singles from Now That I 've Found You : A Collection `` Baby , Now That I 've Found You '' Released : 1995 `` Oh , Atlanta
'' Released : 1995 Professional ratings Review scores Source Rating Allmusic Robert Christgau A − Rolling Stone Now That I 've Found You : A Collection is an
album by Alison Krauss , released February 7 , 1995 . It is a retrospective of the early part of Krauss ' recording career .
</item 8>

<item 9>
Now That I've Found You: A Collection
Close and Personal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Now_That_I%27ve_Found_You:_A_Collection&oldid=808197945 '' Categories : 1995
compilation albums Alison Krauss & Union Station albums Rounder Records compilation albums Hidden categories : Articles with hAudio microformats Music infoboxes
with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 1 November 2017 , at 13 : 34 . About Wikipedia
</item 9>

<item 10>
For What It's Worth
I Hope You Find It '' `` Take It Like a Man '' `` I Walk Alone '' Guest singles `` A Love Like Yours ( Do n't Come Knocking Everyday ) '' `` Dead Ringer for
Love '' `` I Got You Babe '' `` It Ai n't Necessarily So '' `` Love Can Build a Bridge '' `` Più che puoi '' `` Bewitched , Bothered and Bewildered '' Book
Portal Haley Reinhart Discography Videography Songs Studio albums Listen Up ! ( 2012 ) Better ( 2016 ) What 's That Sound ? ( 2017 ) EPs American Idol Season 10
Highlights : Haley Reinhart ( 2011 ) Songs `` Baby , It 's Cold Outside '' `` Free '' `` Ca n't Help Falling in Love '' `` Better '' Concert tours American
Idols LIVE !
</item 10>
=== END ITEM id=04d4704b93fd3326 ===

=== ITEM id=50527e68f582060d ===
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
The Guinness Book of Animal Records . Guinness Publishing . p. 232 . Other sources ( edit ) British Marine Life Study Society -- C. capillata and C. lamarcki
Marine Life Information ( UK ) Marine Biological Laboratory ( Massachusetts ) Pacific Coast jellies External links ( edit ) Wikimedia Commons has media related
to Cyanea capillata . Taxon identifiers Wd : Q522939 ADW : Cyanea_capillata ARKive : cyanea - capillata BioLib : 126478 EoL : 1005690 EPPO : CYNECA Fauna
Europaea : 240643 GBIF : 2264478 iNaturalist : 69838 ITIS : 51671 NCBI : 27804 WoRMS : 135301 Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
Lion's mane jellyfish
Behavior and reproduction ( edit ) Main article : Jellyfish § Life cycle Lion 's mane jellyfish swimming , side view Lion 's mane jellyfish remain mostly very
near the surface , at no more than 20 m depth . Their slow pulsations drive them forwards weakly , so they depend on ocean currents to travel great distances .
The jellyfish are most often spotted during the late summer and autumn , when they have grown to a large size and the currents begin to sweep them to shore .
Like other jellyfish , lion 's manes are capable of both sexual reproduction in the medusa stage and asexual reproduction in the polyp stage . Lion 's mane
jellyfish have four different stages in their year - long lifespan : a larval stage , a polyp stage , an ephyrae stage , and the medusa stage . The female
jellyfish carries its fertilized eggs in its tentacle , where the eggs grow into larva . When the larva are old enough , the female deposits them on a hard
surface , where the larva soon grow into polyps . The polyps begin to reproduce asexually , creating stacks of small creatures called ephyrae .
</item 2>

<item 3>
Lion's mane jellyfish
This was later corrected -- in 1864 , a bootlace worm ( Lineus longissimus ) was found washed up on a Scottish shore that was 55 m ( 180 ft ) long . ( That
claim has its critics : because bootlace worms can easily stretch to several times their natural length , it is possible the worm did not actually grow to be
that length . ) Gallery ( edit ) A lion 's mane jelly capturing a sea gooseberry Lion 's mane jellyfish with full threads visible Lion 's mane jellyfish
contracted Lion 's mane jellyfish in star formation Lion 's mane jellyfish open , showing underside Life - sized model in the Smithsonian Museum of Natural
History , Washington DC , US Small , dead lion 's mane jelly washing up on the beach Blue lion 's mane jellyfish in Askeröfjorden outside Stenungsund , Sweden .
References ( edit ) Jump up ^ Hair Jelly ( Cyanea capillata ) . Australian Venom Research Unit . ^ Jump up to : `` Lion 's Mane Jellyfish -- Reference Library
'' . redOrbit . Retrieved 2010 - 09 - 03 .
</item 3>

<item 4>
Lion's mane jellyfish
Size seems to dictate coloration : larger specimens are a vivid crimson to dark purple while smaller specimens grade to a lighter orange or tan , occasionally
colorless . The bell of the lion 's mane jellyfish is scalloped into eight lobes ( lappets ) , each lobe containing from 70 to 150 tentacles , arranged in four
fairly distinct rows . Along the bell margin is a balance organ at each of the eight indentations between the lobes -- the rhopalium -- which helps the
jellyfish orient itself . From the central mouth extend broad frilly oral arms with many stinging cells . Closer to its mouth , its total number of tentacles
are around 1200 . The long , thin tentacles which emanate from the bell 's subumbrella have been characterised as `` extremely sticky '' ; they also have
stinging cells . The tentacles of larger specimens may trail as long as 30 m ( 100 ft ) or more , with the tentacles of the longest known specimen measured at
37 m ( 120 ft ) in length .
</item 4>

<item 5>
Lion's mane jellyfish
The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles
37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United
States . The lion 's mane jellyfish uses its stinging tentacles to capture , pull in and eat prey such as fish , sea creatures and smaller jellyfish . Contents
( hide ) 1 Taxonomy 2 Description 3 Ecology 4 Behavior and reproduction 5 Sting and human contact 6 In popular culture 7 Gallery 8 References 8.1 Other sources
9 External links Taxonomy ( edit ) Cyanea sp . The taxonomy of the Cyanea species is not fully agreed upon ; some zoologists have suggested that all species
within the genus should be treated as one .
</item 5>

<item 6>
Lion's mane jellyfish
This unusual length -- longer than a blue whale -- has earned it the status of one of the longest known animals in the world . Ecology ( edit ) As coldwater
species , these jellyfish can not cope with warmer waters . The jellyfish are pelagic for most of their lives but tend to settle in shallow , sheltered bays
towards the end of their one - year lifespan . In the open ocean , lion 's mane jellyfish act as floating oases for certain species , such as shrimp ,
medusafish , butterfish , harvestfish , and juvenile prowfish , providing both a reliable source of food and protection from predators . Predators of the lion
's mane jellyfish include seabirds , larger fish such as ocean sunfish , other jellyfish species , and sea turtles . The leatherback sea turtle feeds almost
exclusively on them in large quantities during the summer season around Eastern Canada . The jellyfish themselves feed mostly on zooplankton , small fish ,
ctenophores , and moon jellies .
</item 6>

<item 7>
Lion's mane jellyfish
Considering the size of the species , it is possible that this mass incident was caused by a single specimen . In popular culture ( edit ) The lion 's mane
jellyfish appears in the Sherlock Holmes short story `` The Adventure of the Lion 's Mane '' published in The Case - Book of Sherlock Holmes . Holmes discovers
at the end of the story that the true killer of a professor who died shortly after going swimming ( shouting `` the lion 's mane '' before he succumbed ) was
actually this jellyfish . Suspicion was originally laid upon the professor 's rival in love , until the latter was similarly attacked ( he survived , although
badly stung ) . In the context of the story , it is only because the school professor has a weak heart that he succumbs , as is confirmed by the survival of the
second victim . A photograph widely distributed on the internet appears to show an anomalously large lion 's mane dwarfing a nearby diver by several times . The
photo was subsequently shown to be a hoax . On the popular television program QI the show claimed that the longest animal in the world was the lion 's mane
jellyfish .
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
Lion 's mane jellyfish - wikipedia Lion 's mane jellyfish Jump to : navigation , search Lion 's mane jellyfish Scientific classification Kingdom : Animalia
Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus ,
1758 ) The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its
range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea ,
North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can
not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand .
</item 9>

<item 10>
Lion's mane jellyfish
Two distinct taxa , however , occur together in at least the eastern North Atlantic , with the blue jellyfish ( Cyanea lamarckii Péron & Lesueur , 1810 )
differing in color ( blue , not red ) and smaller size ( 10 -- 20 cm diameter , rarely 35 cm ) . Populations in the western Pacific around Japan are sometimes
distinguished as Cyanea nozakii , or as a subspecies , C. c. nozakii . In 2015 , Russian researchers announced a possible sister species , Cyanea tzetlinii
found in the White Sea , but this has not yet been recognized by other authoritative databases such as WoRMS or ITIS . Description ( edit ) Lion 's mane
jellyfish ( Cyanea capillata ) are named for their showy , trailing tentacles reminiscent of a lion 's mane . They can vary greatly in size : although capable
of attaining a bell diameter of over 2 metres ( 6 ft 7 in ) , those found in lower latitudes are much smaller than their far northern counterparts , with a bell
about 50 centimetres ( 20 in ) in diameter .
</item 10>
=== END ITEM id=50527e68f582060d ===

=== ITEM id=9fe4e1b75b9fd476 ===
Gold sentences and facts:
Sentence 0: There are several major additional characters introduced by the television series and numerous narrative details and the plotline differ radically from the source novel . For example , the planned Nazi pre-emptive nuclear strike on Japan , `` Operation Dandelion , '' is apparently being prevented only by Hitler 's personal refusal to authorise it , leading Heydrich and the faction demanding pre-emptive war to plot the Führer 's assassination . In addition , Hawthorne Abendsen does not appear in the first season of the television version and The Grasshopper Lies Heavy is a series of newsreel films depicting multiple alternative realities rather than a novel ( although this idea may actually be borrowed from Dick 's later novel Valis which features a mysterious film depicting yet another dystopian alternative history of the USA ) . As of the Season 1 finale , these films are being tracked down by SS agents like Blake for dispatch to Hitler for an as - yet - unknown purpose .
  Fact 0.0: The television series introduces several major additional characters.
  Fact 0.1: The narrative details of the television series differ radically from the source novel.
  Fact 0.2: The plotline of the television series differs radically from the source novel.
  Fact 0.3: In the television series, the planned Nazi pre-emptive nuclear strike on Japan, 'Operation Dandelion', is apparently being prevented only by Hitler's personal refusal to authorise it.
  Fact 0.4: Hitler's refusal to authorise Operation Dandelion leads Heydrich and the faction demanding pre-emptive war to plot the Führer's assassination.
  Fact 0.5: Hawthorne Abendsen does not appear in the first season of the television version.
  Fact 0.6: In the television version, The Grasshopper Lies Heavy is a series of newsreel films depicting multiple alternative realities.
  Fact 0.7: In the television version, The Grasshopper Lies Heavy is not a novel.
  Fact 0.8: The idea of The Grasshopper Lies Heavy as newsreel films may actually be borrowed from Dick's later novel Valis.
  Fact 0.9: The novel Valis features a mysterious film depicting yet another dystopian alternative history of the USA.
  Fact 0.10: As of the Season 1 finale, these films are being tracked down by SS agents like Blake.
  Fact 0.11: The films are being tracked down for dispatch to Hitler.
  Fact 0.12: The films are being dispatched to Hitler for an as-yet-unknown purpose.

Retrieved items:
<item 1>
The Man in the High Castle
You can help by adding to it . ( February 2014 ) In The Religion of Science Fiction , Frederick A. Kreuziger explores the theory of history implied by Dick 's
creation of the two alternative realities : Neither of the two worlds , however , the revised version of the outcome of WWII nor the fictional account of our
present world , is anywhere near similar to the world we are familiar with . But they could be ! This is what the book is about . The book argues that this
world , described twice , although differently each time , is exactly the world we know and are familiar with . Indeed , it is the only world we know : the
world of chance , luck , fate . Avram Davidson praised the novel as a `` superior work of fiction '' , citing Dick 's use of the I Ching as `` fascinating '' .
Davidson concluded that `` It 's all here -- extrapolation , suspense , action , art , philosophy , plot , ( and ) character . '' The Man in the High Castle
secured for Dick the 1963 Hugo Award for Best Novel . A new paperback edition of the novel was published in 1992 by Vintage Books .
</item 1>

<item 2>
The Man in the High Castle
The Nazis have drained the Mediterranean to make room for farmland , developed and used the hydrogen bomb , and designed rockets for extremely fast travel
across the world as well as space , having colonized the Moon , Venus , and Mars . The novel is set mostly in San Francisco in the P.S.A. ; here , Chinese
residents first appear in the novel as second - class citizens and black people as slaves . The secondary setting of the novel is the Rocky Mountains States ,
namely the cities of Cañon City , Denver and Cheyenne . Plot summary ( edit ) In 1962 , fifteen years after Imperial Japan and Nazi Germany have won World War
II , Robert `` Bob '' Childan owns an Americana antique shop in San Francisco , California ( located in the Japanese - occupied Pacific States of America ) ,
which is most commonly frequented by the Japanese , who make a fetish of romanticized American cultural artifacts . Childan is contacted by Nobusuke Tagomi , a
high - ranking Japanese trade official , who is seeking a gift to impress a visiting Swedish industrialist named Baynes .
</item 2>

<item 3>
The Man in the High Castle
The Man in the High Castle - wikipedia The Man in the High Castle Jump to : navigation , search This article is about the novel . For the 2015 TV adaptation ,
see The Man in the High Castle ( TV series ) . The Man in the High Castle Cover of first edition ( hardcover ) Author Philip K. Dick Country United States
Language English Genre Alternate history Publisher Putnam Publication date October 1962 Media type Print ( Hardcover & Paperback ) Pages 240 OCLC 145507009
Dewey Decimal 813.54 The Man in the High Castle ( 1962 ) is an alternate history novel by American writer Philip K. Dick . Set in 1962 , fifteen years after an
alternative ending to World War II , the novel concerns intrigues between the victorious Axis Powers -- primarily , Imperial Japan and Nazi Germany -- as they
rule over the former United States , as well as daily life under the resulting totalitarian rule . The Man in the High Castle won the Hugo Award for Best Novel
in 1963 .
</item 3>

<item 4>
The Man in the High Castle
Wegener flies back to Germany where upon landing learns that Reinhard Heydrich ( a member of the anti-Dandelion faction ) has launched a coup against Goebbels ,
possibly installing himself as Chancellor . Meanwhile , Tagomi remains shaken by the shootout and goes to Childan to sell back the gun he used in the fight ;
however , instead , sensing the energy from one of Frink 's jewelery items , Tagomi impulsively buys it from Childan , before undergoing a spiritually intense
ambiguous moment where he momentarily perceives an alternative - history version of San Francisco . Later , Tagomi on a whim forces the German authorities to
release Frink , whom Tagomi has never met and does not know is the maker of the jewelery . Juliana soon has her own spiritual experience when she arrives in
Cheyenne . There , she discovers that Abendsen now lives in a normal house with his family , having left behind the High Castle due to a change of outlook ; he
no longer preoccupies himself with thoughts that he might soon be assassinated .
</item 4>

<item 5>
The Man in the High Castle
Contents ( hide ) 1 Synopsis 1.1 Background 1.2 Plot summary 2 The Grasshopper Lies Heavy 3 Inspirations 4 Reception 5 Adaptations 5.1 Audiobook 5.2 Television
6 Incomplete sequel 7 See also 8 References 9 Further reading 10 External links Synopsis ( edit ) Fictional map of the world after the Axis victory Briefly ,
The Man in the High Castle is a `` fictional picture of a world divided by Germany and Japan , victors of the second World War '' . Background ( edit ) In the
novel 's alternate history , U.S. President Franklin D. Roosevelt was assassinated by Giuseppe Zangara in 1934 leading to the continuation of the Great
Depression and U.S. isolationism during the opening of World War II . Adolf Hitler led Nazi Germany to conquer most of Europe and the Soviet Union , murdering
Jews , Roma , Slavs , and other groups . Meanwhile , Imperial Japan occupied China , before taking control of India and Oceania . The Nazis then helped Italy
conquer most of Africa . As Japan invaded the U.S.
</item 5>

<item 6>
The Man in the High Castle
In the novel , the Italian Empire is a minor power that controls North Africa and the Middle East ; in the series , however , it is shown through maps that
these territories are part of the Nazi Empire , suggesting that either the Italian Empire was annexed after the war or is self - governing within the Reich .
Characters from the book that do appear are in most cases far more fleshed out with deeper and sometimes rather different backstories than their novel originals
. For instance , Wegener is a Standartenführer in the SS , rather than a naval captain ( and oddly there are no German military or naval -- as opposed to SS --
personnel depicted anywhere in the first season ) . Rather than being a member of an organized internal resistance ( and despite his relatively low rank )
Wegener is a close personal confidante of Hitler and his disillusionment with the regime appears to be largely personal . Juliana and Frank are unmarried , but
living together , rather than divorced and separated . Frank has a sister , nephew and niece , although they are killed early in the series , and this propels
him into a more active role in relation to the resistance .
</item 6>

<item 7>
The Man in the High Castle
The pilot episode was released by Amazon Studios on January 15 , 2015 , and was Amazon 's `` most watched pilot ever '' according to Amazon Studios ' vice
president , Roy Price . On February 18 , 2015 , Amazon green - lit the series . The show became available for streaming on November 20 , 2015 . The television
series diverges from the novel in many significant respects . Both the Pacific States of America and the Eastern American puppet state appear to be mere
provinces of the Japanese and German empires without any apparent autonomous ( even quisling ) government institutions whatsoever . The Rocky Mountain States
become an anarchic Neutral Zone . World War II appears to have ended following the destruction of Washington , D.C. with an atomic bomb , rather than a land
invasion as in the book . As for Hitler , while elderly , he is apparently mostly hale in his Season 1 finale appearance , though other characters elsewhere in
the season do reference his supposed physical infirmity .
</item 7>

<item 8>
The Man in the High Castle
In addition , Hawthorne Abendsen does not appear in the first season of the television version and The Grasshopper Lies Heavy is a series of newsreel films
depicting multiple alternative realities rather than a novel ( although this idea may actually be borrowed from Dick 's later novel Valis which features a
mysterious film depicting yet another dystopian alternative history of the USA ) . As of the Season 1 finale , these films are being tracked down by SS agents
like Blake for dispatch to Hitler for an as - yet - unknown purpose . In season two we see a map on John Smith 's wall of the world . From this map we can see
that Japan controls the entirety of the Pacific Ocean and most of the Asian continent , including China , India and half of Russia . Japan is also shown to
control Australia , New Zealand , western Canada , Alaska and , all pacific states in the USA . We also see that that the Japanese Empire controls the Yucatán
Peninsula and South American countries like Chile . In contrast , Nazi Germany is shown as controlling all of Europe , Africa and the Middle East .
</item 8>

<item 9>
The Man in the High Castle
Soon , Joseph Goebbels is announced as the new German Chancellor . Baynes and Tagomi finally meet their Japanese contact as two agents of the Nazi secret police
, the Sicherheitsdienst ( SD ) , close in to arrest Baynes , who is actually revealed to be a Nazi defector named Rudolf Wegener . Wegener warns his contact , a
famed Japanese general , of Operation Dandelion , an upcoming Goebbels - approved plan for the Nazis to surprise - attack the Japanese Home Islands , in order
to obliterate them in one swift stroke . As Frink is elsewhere exposed as a Jew and arrested , Wegener and Tagomi are confronted by the SD agents , both of whom
Tagomi shoots dead with an antique American pistol . Back in Colorado , Joe abruptly changes his appearance and mannerisms before the trip to the High Castle ,
leading Juliana to deduce that he intends to actually murder Abendsen . Joe confirms this , revealing himself to be an undercover Swiss Nazi assassin . Juliana
mortally wounds Joe and drives off to warn Abendsen of the threat to his life .
</item 9>

<item 10>
The Man in the High Castle
In contrast , Nazi Germany is shown as controlling all of Europe , Africa and the Middle East . The German Empire also controls all of the United States and
Canada up to the Rockies in the west and South American countries like Brazil . Germany also controls the western parts of former USSR such as the Ukraine .
There are also buffer zones between the empires . Neither empire seems to have invaded Mexico and this neutral zone continues up through the Rockies and through
Canada . A similar neutral zone is seen dividing Russia in half . Season two also shows two very different lifestyles in the Japanese controlled Pacific states
and the German controlled east . The Germans have colonized the eastern states and are assimilating its inhabitants into Nazi beliefs . The Japanese Empire is
occupying the pacific states versus colonizing them . There is enforced segregation between Japanese and Americans citizens . It is implied that Native
Americans and Pacific Islanders were exterminated by the Germans . Incomplete sequel ( edit ) In a 1976 interview , Dick said he planned to write a sequel novel
to The Man in the High Castle : `` And so there 's no real ending on it . I like to regard it as an open ending .
</item 10>
=== END ITEM id=9fe4e1b75b9fd476 ===

=== ITEM id=66984febf7f3df9b ===
Gold sentences and facts:
Sentence 0: A main vertical stroke is called a stem . The letter m has three , the left , middle , and right stems . The central stroke of an s is called the spine . When the stroke is part of a lowercase and rises above the height of an x ( called the x height ) , it is called an ascender . Letters with ascenders are bdfhkl . A stroke which drops below the baseline is a descender . Letters with descenders are gjpqy . An arching stroke is called a shoulder as in the top of an R or sometimes just an arch , as in hnm . A closed curved stroke is called a bowl in bdopq DOPQ ; B has two bowls . A trailing outstroke , as in jy JQR is called a tail . The inferior diagonal stroke in K is called leg . A short horizontal stroke , as in the center of ef and the middle stroke of EF , is called a bar . When the strokes connect as in A and H or cross strokes as in t is also known as crossbar . A longer horizontal stroke at the top or bottom , as in ET , is called an arm . The bottom of the two - story g is called a loop ; the very short stroke at the top is called the ear . ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has one vertex .
  Fact 0.0: A main vertical stroke is called a stem.
  Fact 0.1: The letter m has three stems: the left stem, the middle stem, and the right stem.
  Fact 0.2: The central stroke of an s is called the spine.
  Fact 0.3: When a stroke is part of a lowercase letter and rises above the height of an x (called the x height), it is called an ascender.
  Fact 0.4: The letters with ascenders are b, d, f, h, k, and l.
  Fact 0.5: A stroke which drops below the baseline is called a descender.
  Fact 0.6: The letters with descenders are g, j, p, q, and y.
  Fact 0.7: An arching stroke is called a shoulder, as in the top of an R.
  Fact 0.8: An arching stroke is sometimes just called an arch, as in h, n, and m.
  Fact 0.9: A closed curved stroke is called a bowl, as in b, d, o, p, q, D, O, P, and Q.
  Fact 0.10: The letter B has two bowls.
  Fact 0.11: A trailing outstroke, as in j, y, J, Q, and R, is called a tail.
  Fact 0.12: The inferior diagonal stroke in K is called a leg.
  Fact 0.13: A short horizontal stroke, as in the center of e and f and the middle stroke of E and F, is called a bar.
  Fact 0.14: When strokes connect, as in A and H, or cross, as in t, the stroke is also known as a crossbar.
  Fact 0.15: A longer horizontal stroke at the top or bottom, as in E and T, is called an arm.
  Fact 0.16: The bottom of the two-story g is called a loop.
  Fact 0.17: The very short stroke at the top of the two-story g is called the ear.
  Fact 0.18: The letters i and j each have a dot, jot, or tittle.
  Fact 0.19: Angles of strokes are called apices if at the top.
  Fact 0.20: Angles of strokes are called vertices if at the bottom.
  Fact 0.21: The letter w has one apex and two vertices.
  Fact 0.22: The letter v has one vertex.

Retrieved items:
<item 1>
Typeface anatomy
ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has
one vertex . The font shown in the example is stressed : strokes have varying widths . In this example , the stroke at the top of the g is thinner at the top
and bottom than on the sides -- a vertical stress . Terminals ( edit ) The terminals ( ends ) of instrokes and outstrokes often end in serif or sans - serif . A
seriffed terminal may be described as a wedge , bulbous , teardrop , slab , etc. , depending on the design of the type . Sorting typefaces can be practical
based on its look . A clue feature is to define the serif style . Some designs also have spurs , which are smaller than serifs and appear on angles rather than
at a terminal , as on e or G . Space ( edit ) Areas of negative space ( white space ) formed by straight or curved strokes are called counters . Closed counters
are found in abdegopq ABDOPQR , and open counters in acefhmnrstu .
</item 1>

<item 2>
Typeface anatomy
`` Typeface Mechanics : 001 '' . Frere - Jones Type . Retrieved 17 April 2016 . Frere - Jones , Tobias . `` Typeface Mechanics : 002 '' . Frere - Jones Type .
Retrieved 17 April 2016 . Coles , Stephen . `` The Anatomy of Type '' . Blog for Coles ' 2012 book The Anatomy of Type : A Graphic Guide to 100 Typefaces . ``
Glossary '' . FontShop . Monotype . ( hide ) Typography terminology Page Canons of page construction Column Even working Margin Page numbering Pagination Pull
quote Recto and verso Paragraph Alignment Justification Leading River Sentence spacing Widows and orphans Character Typeface anatomy Counter Diacritics Dingbat
Glyph Initial Kerning Letter - spacing Ligature Subscript and superscript Swash Text figures Tittle Capitalization ALL CAPS Camel Case Letter case Petite caps
Small caps Visual distinction Italics Oblique Bold Color Underline Blackboard bold Blackletter Infɑnt Vertical aspects Ascender Baseline Cap height Descender
Median Overshoot x-height Classifications Roman type Antiqua ( old style ) Didone ( modern ) Sans - serif Script Serif Slab serif Transitional Reverse
</item 2>

<item 3>
Typeface anatomy
Harper Design , 2013 . ^ Jump up to : Dean , Paul ( 27 March 2008 ) . `` eXtreme Type Terminology . Part 2 : Anatomy of a Letterform '' . I Love Typography .
Further reading ( edit ) Gaskell , Philip ( January 1976 ) . `` A nomenclature for the letterforms of roman type '' . Visible Language . 10 ( 1 ) : 41 -- 51 .
External links ( edit ) Wikimedia Commons has media related to Examples of typography terms . Armannsson , Sigurdur ( 6 July 2009 ) . `` Wallpaper : Font
Anatomy '' . Font.is . Full - size image ( 1920 × 1200 ) Dean , Paul ( 2008 ) . eXtreme Type Terminology -- Part 1 , Part 2 , Part 3 , Part 4 and Part 5 Devroye
, Luc . `` Hoefler & Frere - Jones '' . Retrieved 21 January 2015 . Frere - Jones , Tobias . `` Typeface Mechanics : 001 '' . Frere - Jones Type .
</item 3>

<item 4>
Typeface anatomy
This made it possible to mix typefaces from completely different genres such as sans - serifs and serifs and have the cap height , baseline and linespacing
match perfectly , something not possible with most digital fonts . It even allowed mixing of different sizes of type with a consistent baseline . It however had
the disadvantage of often forcing typefaces to be issued with cropped descenders compared to historical typefaces , to allow tight linespacing . A `` script
line '' or `` art line '' was used for more delicate fonts with long descenders . Titling capitals , meanwhile , were issued taking up the whole space of the
metal type area , with no room for descenders . See also ( edit ) Type design Type foundry Typography Typesetting References ( edit ) Jump up ^ Studer , Anton (
29 February 2016 ) . `` Is What I See What I Get ? -- Math & Optics in Type Design '' . Typographica . Retrieved 17 April 2016 . ^ Jump up to : Carter , Bob ;
Day , Ben ; Meggs , Philip ( 2007 ) . Typographic Design : Form and Communication .
</item 4>

<item 5>
Typeface anatomy
Typographers also speak of an instroke , where one starts writing the letter , as at the top of acf , and an outstroke , where the pen leaves off , as at the
bottom of cejkty . Typefaces are born from the struggle between rules and results . Squeezing a square about 1 % helps it look more like a square ; to appear
the same height as a square , a circle must be measurably taller . The two strokes in an X are n't the same thickness , nor are their parallel edges actually
parallel ; the vertical stems of a lowercase alphabet are thinner than those of its capitals ; the ascender on ad is n't the same length as the descender on ap
, and so on . For the rational mind , type design can be a maddening game of drawing things differently in order to make them appear the same . Hoefler & Frere
- Jones A main vertical stroke is called a stem . The letter m has three , the left , middle , and right stems . The central stroke of an s is called the spine
.
</item 5>

<item 6>
Typeface anatomy
Typeface Anatomy - wikipedia Typeface Anatomy Jump to : navigation , search Typographic parts of a glyph : 1 ) x-height ; 2 ) ascender line ; 3 ) apex ; 4 )
baseline ; 5 ) ascender ; 6 ) crossbar ; 7 ) stem ; 8 ) serif ; 9 ) leg ; 10 ) bowl ; 11 ) counter ; 12 ) collar / link / neck ; 13 ) loop ; 14 ) ear ; 15 ) tie
; 16 ) horizontal bar ; 17 ) arm ; 18 ) vertical bar ; 19 ) cap height ; 20 ) descender line . Typeface anatomy describes the graphic elements that make up
printed letters in a typeface . Contents ( hide ) 1 Strokes 2 Terminals 3 Space 4 Proportions 5 Metal type era 6 See also 7 References 8 Further reading 9
External links Strokes ( edit ) The strokes are the components of a letterform . Strokes may be straight , as in klvwxz , or curved , as in cos . If straight ,
they may be horizontal , vertical , or diagonal ; if curved , open or closed .
</item 6>

<item 7>
Graph query row:
entity: crossbar
relationship: PART_OF
related: glyph
</item 7>

<item 8>
Typeface anatomy
The close counter in e is also named eye . Angles of white space , as in w , are corners ( w has three corners ) ; the term is not used for angles of strokes .
The small corner formed by a serif , whether curved or angular , is called the serif bracket . Proportions ( edit ) A subtle change in proportion impacts weight
, perception , measure , and legibility . The letterform height compared to its stroke width modifies the aspect ; this slight change in weight sometimes helps
to create emphasis . The disparity between thick and thin strokes , known as stress , alters optical perception . As an example , the initial san serif
typefaces used a constantly thin stroke . The technological advances allow drawing thinner strokes . Condense type occupies less space than expanded type , then
the text written on a page can be reduced to half a page. The capline and x-height ratio improve or decrease word legibility . Metal type era ( edit ) During
the late metal type period , many fonts ( particularly in American typefounding ) were issued to `` common line '' . This meant that they were made to
standardised proportions , so that fonts of different typefaces could be mixed with no difficulty .
</item 8>

<item 9>
Typeface anatomy
( old style ) Didone ( modern ) Sans - serif Script Serif Slab serif Transitional Reverse - contrast Blackletter type Fraktur Rotunda Schwabacher Textualis
Gaelic type Insular Uncial Specialist Record type Punctuation Dashes Hanging punctuation Hyphen - minus Hyphenation Prime mark Quotation mark Typesetting
Calligraphy etaoin shrdlu Font computer monospaced Font catalog Letterpress Lorem ipsum Microtypography Movable type Pangram Phototypesetting Punchcutting Type
color Type design Typeface Microprint list Typographic units Agate Cicero Em En Figure space Measure Paren space Pica Point traditional point - size names Thin
space Digital typography Character encoding Font formats Hinting Rasterization Typesetting software Typographic features Web typography Related Intentionally
blank page Style guide Type foundry Commons Portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Typeface_anatomy&oldid=835865029 '' Categories :
Typography Talk Contents About Wikipedia Polski Русский Edit links This page was last edited on 11 April 2018 , at 07 : 14 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 9>

<item 10>
Typeface anatomy
The central stroke of an s is called the spine . When the stroke is part of a lowercase and rises above the height of an x ( called the x height ) , it is
called an ascender . Letters with ascenders are bdfhkl . A stroke which drops below the baseline is a descender . Letters with descenders are gjpqy . An arching
stroke is called a shoulder as in the top of an R or sometimes just an arch , as in hnm . A closed curved stroke is called a bowl in bdopq DOPQ ; B has two
bowls . A trailing outstroke , as in jy JQR is called a tail . The inferior diagonal stroke in K is called leg . A short horizontal stroke , as in the center of
ef and the middle stroke of EF , is called a bar . When the strokes connect as in A and H or cross strokes as in t is also known as crossbar . A longer
horizontal stroke at the top or bottom , as in ET , is called an arm . The bottom of the two - story g is called a loop ; the very short stroke at the top is
called the ear .
</item 10>
=== END ITEM id=66984febf7f3df9b ===

=== ITEM id=45d87b1914ff786e ===
Gold sentences and facts:
Sentence 0: Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 - 08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco 49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 , 1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All - American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9 Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of Fame
  Fact 0.0: Deion Sanders is pictured in 2011.
  Fact 0.1: Deion Sanders' jersey numbers are No. 21 and No. 37.
  Fact 0.2: Deion Sanders' position is cornerback.
  Fact 0.3: Deion Sanders was born on August 9, 1967.
  Fact 0.4: Deion Sanders is listed as age 50.
  Fact 0.5: Deion Sanders was born in Fort Myers, Florida.
  Fact 0.6: Deion Sanders is 6 ft 1 in (1.85 m) tall.
  Fact 0.7: Deion Sanders weighs 198 lb (90 kg).
  Fact 0.8: Deion Sanders' high school is North Fort Myers (FL).
  Fact 0.9: Deion Sanders' college is Florida State.
  Fact 0.10: Deion Sanders was picked in round 1, pick 5 of the 1989 NFL Draft.
  Fact 0.11: Deion Sanders played for the Atlanta Falcons from 1989 to 1993.
  Fact 0.12: Deion Sanders played for the San Francisco 49ers in 1994.
  Fact 0.13: Deion Sanders played for the Dallas Cowboys from 1995 to 1999.
  Fact 0.14: Deion Sanders played for the Washington Redskins in 2000.
  Fact 0.15: Deion Sanders played for the Baltimore Ravens from 2004 to 2005.
  Fact 0.16: Deion Sanders is a 2-time Super Bowl champion, for Super Bowls XXIX and XXX.
  Fact 0.17: Deion Sanders is an 8-time Pro Bowl selection (1991-1994, 1996-1999).
  Fact 0.18: Deion Sanders is an 8-time First-team All-Pro (1991-1994, 1996-1999).
  Fact 0.19: Deion Sanders was NFL Defensive Player of the Year in 1994.
  Fact 0.20: Deion Sanders is named to the NFL 1990s All-Decade Team.
  Fact 0.21: Deion Sanders has an Atlanta Falcons Ring of Honor entry.
  Fact 0.22: Deion Sanders won the Jim Thorpe Award in 1988.
  Fact 0.23: Deion Sanders is a 2-time Unanimous All-American (1987, 1988).
  Fact 0.24: No. 2 is retired by the Florida State Seminoles.
  Fact 0.25: Deion Sanders' career NFL interceptions total 53.
  Fact 0.26: Deion Sanders' career NFL defensive touchdowns total 10.
  Fact 0.27: Deion Sanders' career NFL return touchdowns total 9.
  Fact 0.28: Deion Sanders' career NFL receptions total 60.
  Fact 0.29: Deion Sanders' career NFL receiving yards total 784.
  Fact 0.30: Deion Sanders' career NFL receiving touchdowns total 3.
  Fact 0.31: The page links to player stats at NFL.com.
  Fact 0.32: The page links to player stats at PFR.
  Fact 0.33: The page links to the Pro Football Hall of Fame.
  Fact 0.34: The page links to the College Football Hall of Fame.

Retrieved items:
<item 1>
Deion Sanders
In 2014 , he debuted his new show , the Deion Family Playbook . Contents ( hide ) 1 Early life 1.1 High school 1.2 College career 2 Professional baseball career
3 Professional football career 3.1 Draft and Atlanta Falcons 3.2 San Francisco 49ers 3.3 Dallas Cowboys 3.4 Washington Redskins 3.5 Baltimore Ravens 4 Legacy 5
Media appearances and pop culture fame 6 Other business and entertainment ventures 7 Discography 8 Personal life 9 See also 10 References 11 External links
Early life ( edit ) High school ( edit ) Sanders was born in Fort Myers , Florida . He attended North Fort Myers High School in North Fort Myers , and was a
letterman and All - State honoree in football , basketball and baseball . In 1985 , Sanders was named to the Florida High School Association All - Century Team
which selected the top 33 players in the 100 - year history of high school football in the state . The Kansas City Royals selected Sanders out of North Fort
Myers High School in the sixth round of the 1985 Major League Baseball draft . He did not sign with the Royals .
</item 1>

<item 2>
Deion Sanders
Deion Sanders - wikipedia Deion Sanders Jump to : navigation , search Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 -
08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High
school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco
49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl
champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 ,
</item 2>

<item 3>
Deion Sanders
He played in the National Football League ( NFL ) for the Atlanta Falcons , the San Francisco 49ers , the Dallas Cowboys , the Washington Redskins and the
Baltimore Ravens , winning the Super Bowl with both the 49ers and the Cowboys . An outfielder in baseball , he played professionally for the New York Yankees ,
the Atlanta Braves , the Cincinnati Reds and the San Francisco Giants , and participated in the 1992 World Series with the Braves . He attended Florida State
University , where he was recognized as a two - time All - American in football , and also played baseball and ran track . Sanders founded the Prime Prep
Academy charter school in 2012 and has coached at the school . The school was closed January 30 , 2015 , due to financial insolvency . In 2014 , he debuted his
new show , the Deion Family Playbook .
</item 3>

<item 4>
Deion Sanders
Sanders amassed 7,838 all - purpose yards and scored 22 touchdowns : nine interception returns , six punt returns , three kickoff returns , three receiving ,
and one fumble recovery . His 19 defensive and return touchdowns was an NFL record ( now held by Devin Hester ) . In the postseason , Sanders added 5 more
interceptions , as well as 3 receptions for 95 yards , 4 carries for 39 yards , and two touchdowns ( one rushing and one receiving ) . He was selected to eight
Pro Bowls and won the NFL Defensive Player of the Year Award in 1994 . College Football News named Sanders # 8 in its list of 100 Greatest College Football
Players of All - Time . The Sporting News named Sanders # 37 in their Top 100 Football Players of the Century released in 1999 . ESPN named Sanders # 74 in its
list of the 100 Great Athletes of the Century released in 1999 . NFL.com named Sanders # 34 on NFL 's Top 100 list released in late 2010 On November 11 , 2010 ,
Sanders was inducted into the Atlanta Falcons ' Ring of Honor .
</item 4>

<item 5>
Deion Sanders
NFL . National Football League . Retrieved 28 January 2017 . Jump up ^ `` Deion Sanders HOF Speech '' . Video Entry . Retrieved October 19 , 2011 . Jump up ^
Kallestad , Brent . `` Deion Changes the Rules Again : Pro football : This time , however , it 's for better education of athletes with new standards in Florida
. '' . Los Angeles Times . Retrieved 29 January 2017 . Jump up ^ Trex , Ethan . `` Two Rules Named After Deion Sanders '' . mental_floss . Mental Floss , Inc .
Retrieved 29 January 2017 . Jump up ^ `` NFL Videos : 2011 HOF : Deion Sanders '' . Nfl.com . August 6 , 2011 . Retrieved August 16 , 2012 . Jump up ^ Frohman ,
Jimmy . `` 11 Most Awesome NFL Player Nicknames '' . Phactual . Phactual , LLC . Retrieved 24 January 2017 . Jump up ^ `` CES ProNews Flashes ! '' .
</item 5>

<item 6>
Deion Sanders
1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All -
American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9
Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of
Fame Deion Luywnn Sanders Sr. ( / ˈdiːɒn / ; born August 9 , 1967 ) , nicknamed Primetime , is a former American football and baseball player who works as an
analyst for CBS Sports and the NFL Network . He was inducted into the Pro Football Hall of Fame in 2011 . Sanders played football primarily at cornerback , but
also as a kick returner , punt returner and occasionally wide receiver .
</item 6>

<item 7>
Deion Sanders
In 1995 , he signed with the Dallas Cowboys for a minimum yearly base salary and a nearly $13 million signing bonus in an attempt to circumvent the NFL 's
salary cap . This caused the NFL to institute its own ' Deion Sanders rule ' whereby a prorated portion of a player 's signing bonus counted against the salary
cap . Media appearances and pop culture Fame ( edit ) See also : Leon Sandcastle Sanders became known for sporting a `` do - rag '' or bandana and for his high
- stepping into the end zone followed by his touchdown dance celebrations . At the end of his Hall of Fame speech , he put a bandana on his bust . His `` Prime
Time '' nickname was given to him by a friend and high - school teammate , Florida Gators defensive back Richard Fain . The two played pickup basketball games
together during the prime time television hour , and Sanders ' athletic display during those games earned him the nickname . His other nickname is `` Neon Deion
'' . Following his first Super Bowl victory with the San Francisco 49ers , Sanders hosted Saturday Night Live , broadcast on February 18 , 1995 .
</item 7>

<item 8>
Deion Sanders
After getting into a scuffle with his former Falcon teammate Andre Rison , Sanders intercepted a pass from quarterback Jeff George and returned it 93 yards
while mockingly staring down the entire Falcons sideline before high - stepping into the end zone . Sanders was later voted the 1994 NFL Defensive Player of the
Year and recorded an interception in the 49ers 49 -- 26 victory over the San Diego Chargers in Super Bowl XXIX , earning him his first championship ring .
Dallas Cowboys ( edit ) Following his successful season with the 49ers , Sanders , along with his agent Eugene Parker , courted numerous teams in need of a
cornerback . The several teams in the `` Deion Sweepstakes '' , as it was called by the media , were the Philadelphia Eagles , Oakland Raiders , Miami Dolphins
, New Orleans Saints , San Francisco 49ers and the Dallas Cowboys , who had lost their starting cornerback Kevin Smith to injury for the rest of the season .
</item 8>

<item 9>
Deion Sanders
Sanders also occasionally lined up with his team 's offense . During the 1996 season , Sanders skipped the baseball season , concentrating on football , and
attended the first NFL training camp of his career to better familiarize himself with the nuances of the wide receiver position . He became only the second two
- way starter ( after the Cardinals ' Roy Green ) in the NFL since Chuck Bednarik for the first half of the season due . Sanders is the only man to play in both
a Super Bowl and a World Series , to hit an MLB home run and score an NFL touchdown in the same week , and to have both a reception and an interception in the
Super Bowl . He is one of two players to score an NFL touchdown six different ways . During his career , Sanders intercepted 53 passes for 1,331 yards ( a 25.1
yards per return average ) , recovered four fumbles for 15 yards , returned 155 kickoffs for 3,523 yards , gained 2,199 yards on 212 punt returns , and caught
60 passes for 784 yards .
</item 9>

<item 10>
Deion Sanders
On February 5 , 2011 , Sanders was announced as a Pro Football Hall of Fame inductee in his first year of eligibility . On May 17 , 2011 , Sanders was announced
as a College Football Hall of Fame inductee . On August 6 , 2011 , Sanders was inducted into the Pro Football Hall of Fame . Sanders also had a rushing TD in
the playoffs ( against the Philadelphia Eagles in January 1996 ) . This makes him ( including post season ) one of only two players in NFL history ( Bill Dudley
being the other ) to score a touchdown six different ways ( interception return , punt return , kickoff return , receiving , rushing , and a fumble recovery ) .
On February 6 , 2011 , at Super Bowl XLV in Arlington , Texas , Sanders performed the pre-game coin toss . Sanders did not attend classes or take final exams
during the fall semester of his senior year at Florida State , yet played in the Sugar Bowl against Auburn . This caused the Florida State Legislature to create
the ' Deion Sanders rule ' , whereby a football athlete at any state university could not play in a bowl game if he failed to successfully complete the previous
semester .
</item 10>
=== END ITEM id=45d87b1914ff786e ===

=== ITEM id=e99e8708425e3616 ===
Gold sentences and facts:
Sentence 0: On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . Shaftesbury Films confirmed that the series will not be returning to Super Channel or Chiller for a second season . However , Netflix acquired the rights to the second season . The second season was released via Netflix on October 17 , 2017 .
  Fact 0.0: Slasher began filming a second season on May 1, 2017.
  Fact 0.1: Slasher filmed its second season in Orangeville, Ontario, Canada.
  Fact 0.2: Shaftesbury Films confirmed that Slasher will not return to Super Channel or Chiller for a second season.
  Fact 0.3: Netflix acquired the rights to the second season of Slasher.
  Fact 0.4: The second season of Slasher was released via Netflix on October 17, 2017.
Sentence 1: The first season , retroactively subtitled The Executioner , was produced in association with the Canadian network Super Channel , Slasher was the first original series by U.S. TV channel Chiller , which premiered the series on Friday , March 4 , 2016 , at 9 : 00 pm EST . Super Channel aired the show 's Canadian premiere on April 1 , 2016 . The series first season , which centered on a mysterious figure billed as `` The Executioner '' terrorizing the fictional town of Waterbury , Canada , was filmed between July and October 2015 in Sudbury , Parry Sound , and Sault Ste . Marie , Ontario .
  Fact 1.0: The first season of Slasher, retroactively subtitled The Executioner, was produced in association with the Canadian network Super Channel.
  Fact 1.1: Slasher was the first original series by U.S. TV channel Chiller.
  Fact 1.2: Chiller premiered the series on Friday, March 4, 2016, at 9:00 pm EST.
  Fact 1.3: Super Channel aired the show's Canadian premiere on April 1, 2016.
  Fact 1.4: The first season of Slasher centered on a mysterious figure billed as 'The Executioner'.
  Fact 1.5: The Executioner terrorized the fictional town of Waterbury, Canada.
  Fact 1.6: The first season was filmed between July and October 2015.
  Fact 1.7: The first season was filmed in Sudbury, Parry Sound, and Sault Ste. Marie, Ontario.

Retrieved items:
<item 1>
Slasher (TV series)
Not wanting to have the show 's killer be `` a mythological creature '' ( as he feels the killers in most slasher films do not have much mystery surrounding
them ) , Martin also uses elements of the traditional whodunit in Slasher : The show 's characters , many of whom have mysterious backgrounds -- and their own
reasons for possibly being the killer -- are featured , explored , and eliminated from consideration , one by one ( either through death or the natural
deductive process ) , until the `` all too human '' killer and their motivations for their actions are revealed . Plot ( edit ) Season 1 : the Executioner (
2016 ) ( edit ) Slasher 's 8 - episode first season centers around Sarah Bennett who , with husband Dylan , moves back to the town of her birth , Waterbury ( a
fictional town in Canada ) , and into her parents ' former home . It was in that house on Halloween Night 1988 where both of her parents , Bryan and Rachel ,
were murdered . Rachel was pregnant with Sarah at the time of her murder , with police discovering the killer holding Rachel 's newborn baby after the slayings
.
</item 1>

<item 2>
Slasher (TV series)
Slasher ( TV series ) - wikipedia Slasher ( TV series ) Jump to : navigation , search Slasher Genre Anthology Horror Slasher Drama Created by Aaron Martin
Starring Katie McGrath Brandon Jay McLaren Steve Byers Patrick Garrow Dean McDermott Leslie Hope Lovell Adams - Gray Jim Watson Christopher Jacot Paulino Nunes
Ty Olsson Joanne Vannicola Sebastian Pigott Madison Cheeatow Melinda Shankar Kaitlyn Leeb Rebecca Liddiard Paula Brancati Composer ( s ) Shawn Pierce Country of
origin Canada Original language ( s ) English No. of seasons No.
</item 2>

<item 3>
Slasher (TV series)
A first trailer for the series would be released on November 26 , 2015 . Unlike most TV series that film their episodes in order , Slasher , under the direction
of Craig David Wallace , was shot as if it were a `` super-sized '' movie : Scenes from multiple episodes were shot at the same time , with the availability of
locations and cast being factored in . The out - of - order schedule allowed the actors to know of their characters ' fates , especially those who had to film
their death scenes one day but return later to film earlier scenes as necessary . As an example of this , Martin cited Mark Ghanimé 's first day on set , when
his character , Justin Faysal , was laid out in a casket for a scene early in Slasher 's third episode ( Justin 's death , which took place in Episode 2 , would
be filmed later on ) . On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . Shaftesbury Films confirmed that the series
will not be returning to Super Channel or Chiller for a second season .
</item 3>

<item 4>
Slasher (TV series)
They 're all bright eyed , bushy tailed , ready to live an alternative lifestyle in the midst of nature . Quite different from how it is now . Especially with a
chainsaw revving in the distance . Renée finds a box of things from 2012 marked Camp Motega . She looks through it and comes across the files of the group of
friends , from Andi ( Rebecca Liddiard ) to Talvinder to Dawn and the rest . And then we see Wren . Outside the killer , covered in blood , arrives on
snowmobile . He runs Mark down hard . Then he puts a couple bullets into Renée as she runs away . She does n't make it very far . The killer takes the file from
her , then we hear the sounds of him running her over , the wet sounds of the blood , her last scream . 15 7 `` Dawn of the Dead '' Felipe Rodriguez Amanda
Fahey October 17 , 2017 ( 2017 - 10 - 17 ) Five years ago , Talvinder ( Melinda Shankar ) and Dawn ( Paula Brancati ) are close .
</item 4>

<item 5>
Slasher (TV series)
Super Channel aired the show 's Canadian premiere on April 1 , 2016 . The series first season , which centered on a mysterious figure billed as `` The
Executioner '' terrorizing the fictional town of Waterbury , Canada , was filmed between July and October 2015 in Sudbury , Parry Sound , and Sault Ste . Marie
, Ontario . On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . The licensing rights for the second season was acquired
by Netflix . The season was released as a `` Netflix Original '' title in various countries on October 17 , 2017 . The second season titled Slasher : Guilty
Party is centered on a group of former summer camp counselors who return to an isolated campground in order to bury their sins of the past , before being
targeted , one by one , by an unknown killer .
</item 5>

<item 6>
Slasher (TV series)
However , Netflix acquired the rights to the second season . The second season was released via Netflix on October 17 , 2017 . Reception ( edit ) Slasher has
received positive reviews . Zap2it called the series `` a whole lot of fun '' and `` something for everyone , '' praising the series ' anthology nature , its
cast of characters , storyline , plot twists , bloody violence , and even the series - within - the - series Falcon Husbandry ( shown as a favorite of Robin and
Justin 's in Episode 2 ) . Bloody Disgusting awarded the show four skulls out of five , praising Katie McGrath as a great `` protagonist and possible final girl
'' and the series ' decision to feature an adult cast , rather than teenagers , with well - developed characters and a `` decidedly classic '' presentation . On
the occasion of Slasher 's Super Channel premiere , The Globe and Mail 's John Doyle , while remarking that it `` is no masterpiece of horror , nor was it meant
to be , '' called the show `` very well - crafted , '' praising its `` exceptional cast '' and tight pacing , and noting fans of gory horror will appreciate its
bloody scenes .
</item 6>

<item 7>
Slasher (TV series)
of seasons No. of episodes 16 ( list of episodes ) Production Executive producer ( s ) Aaron Martin Greg Phillips Jonathan Ford Christina Jennings Scott Garvie
Location ( s ) Ontario , Canada Editor ( s ) Erin Deck Camera setup Single - camera Running time 46 - 53 minutes Production company ( s ) Shaftesbury Films
Super Channel ( season 1 ) TVA Group ( season 2 ) Distributor Content Media Corporation Release Original network Super Channel ( Canada , season 1 ) Chiller (
USA , season 1 ) Netflix ( season 2 ) Original release March 4 , 2016 ( 2016 - 03 - 04 ) -- present External links Website www.netflix.com/title/80105452
Slasher is a Canadian television horror anthology series created by Aaron Martin . The first season , retroactively subtitled The Executioner , was produced in
association with the Canadian network Super Channel , Slasher was the first original series by U.S. TV channel Chiller , which premiered the series on Friday ,
March 4 , 2016 , at 9 : 00 pm EST . Super Channel aired the show 's Canadian premiere on April 1 , 2016 .
</item 7>

<item 8>
Slasher (TV series)
Contents ( hide ) 1 Background 2 Plot 2.1 Season 1 : The Executioner ( 2016 ) 2.2 Season 2 : Guilty Party ( 2017 ) 3 Cast and characters 3.1 Recurring cast
members 3.2 Season 1 : The Executioner 3.2. 1 Main 3.2. 2 Recurring 3.3 Season 2 : Guilty Party 4 Episodes 4.1 Season 1 : The Executioner ( 2016 ) 4.2 Season 2
: Guilty Party ( 2017 ) 5 Development and production 6 Reception 6.1 Awards and accolades 7 International broadcast 8 References 9 External links Background (
edit ) Slasher employs a season - long anthology format , with an overarching mystery storyline that will be resolved upon the season 's conclusion .
</item 8>

<item 9>
Slasher (TV series)
The Futon Critic . July 28 , 2016 . Retrieved March 6 , 2016 . ^ Jump up to : `` Lights , camera , action ... ' Slasher ' TV series begins filming in
Orangeville '' . Orangeville Banner . May 1 , 2017 . Retrieved May 2 , 2017 . ^ Jump up to : `` Slasher returns for a second season '' . Playback Online . May 9
, 2017 . Retrieved September 29 , 2017 . Jump up ^ Moore , Kasey ( September 20 , 2017 ) . `` October 2017 New Netflix Releases '' . WhatsonNetlix . Retrieved
September 20 , 2017 . Jump up ^ Urqyhart - White , Alaina ( October 17 , 2017 ) . `` Is ' Slasher 2 : Guilty Party ' A True Story ? The Netflix Series Honors
Some Legendary Horror Classics '' . Bustle . Retrieved October 18 , 2017 . ^ Jump up to : Patrick Hipes .
</item 9>

<item 10>
Slasher (TV series)
Season 2 : Guilty Party ( 2017 ) ( edit ) Slasher 's 8 - episode second season centers around a horrific secret that a group of former summer camp counselors
have buried long ago . Due to a changing circumstance , they must return to their resort in the dead of winter to retrieve the body of Talvinder , whom they
murdered while they were camp counsellors , five years earlier . They get surprised when they ca n't find the body at the campsite . Later , the group start
dying one by one at the hands of a mysterious person wearing a parka , which also alerts the second group of counselors at the resort . Eventually , it 's
revealed that Judith is the killer , and another character , Wren , is actually Owen `` Wren '' Turnbull , her son who committed suicide after being framed for
Talvinder 's murder , and is a figment of Judith 's imagination . Owen wrote a letter to his mother , asking her to get justice for him before hanging himself .
</item 10>
=== END ITEM id=e99e8708425e3616 ===
