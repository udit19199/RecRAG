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

=== ITEM id=e060aa54aef17fde ===
Gold sentences and facts:
Sentence 0: `` For What It 's Worth ( Stop , Hey What 's That Sound ) '' ( often referred to as simply `` For What It 's Worth '' ) is a song written by Stephen Stills . It was performed by Buffalo Springfield , recorded on December 5 , 1966 , and released as a single on Atco Records in January 1967 . The single peaked at number seven on the Billboard Hot 100 chart . This song is currently ranked number 63 on Rolling Stone 's list of The 500 Greatest Songs of All Time as well as the eighth best song of 1967 by Acclaimed Music .
  Fact 0.0: 'For What It's Worth (Stop, Hey What's That Sound)' is often referred to as simply 'For What It's Worth'.
  Fact 0.1: 'For What It's Worth' is a song.
  Fact 0.2: 'For What It's Worth' was written by Stephen Stills.
  Fact 0.3: 'For What It's Worth' was performed by Buffalo Springfield.
  Fact 0.4: 'For What It's Worth' was recorded on December 5, 1966.
  Fact 0.5: 'For What It's Worth' was released as a single on Atco Records in January 1967.
  Fact 0.6: The 'For What It's Worth' single peaked at number seven on the Billboard Hot 100 chart.
  Fact 0.7: 'For What It's Worth' is currently ranked number 63 on Rolling Stone's list of The 500 Greatest Songs of All Time.
  Fact 0.8: 'For What It's Worth' is the eighth best song of 1967 according to Acclaimed Music.

Retrieved items:
<item 1>
For What It's Worth
Against the background of these civil disturbances , Stills recorded the song on December 5 , 1966 . Production ( edit ) Stills said in an interview that the
name of the song came about when he presented it to the record company executive Ahmet Ertegun who signed Buffalo Springfield to the Atlantic Records - owned
ATCO label . He said : `` I have this song here , for what it 's worth , if you want it . '' Another producer , Charlie Greene , claims that Stills first said
the above sentence to him , but credits Ahmet Ertegun with subtitling the single `` Stop , Hey What 's That Sound '' so that the song would be more easily
recognized . The song was recorded on December 5 , 1966 at Columbia Studios , Hollywood . Tom Dowd claims he mixed the song at Atlantic 's studio in New York ,
though this is disputed by others . He did take part in the production of Cher 's version of the song in 1969. . The song is distinguished with the use of the
guitar harmonics technique and sound . Legacy ( edit ) The song quickly became a well - known protest song .
</item 1>

<item 2>
Closing Time (album)
`` Closing Time '' ( instrumental ) 4 : 20 Total length : 45 : 46 Personnel ( edit ) Musicians ( edit ) Tom Waits -- vocals , piano , guitar , harmonium ,
harpsichord , celeste Delbert Bennett -- trumpet Shep Cooke -- guitar , backing vocals on `` Old Shoes ( & Picture Postcards ) '' Peter Klimes -- guitar , pedal
steel guitar on `` Rosie '' Bill Plummer -- double bass John Seiter -- drums ; backing vocals on `` Ol ' 55 '' and `` Rosie '' Guest musicians ( edit ) Arni
Egilsson -- bass guitar on `` Closing Time '' ( Instrumental ) Jesse Ehrlich -- cello on `` Martha '' Tony Terran -- trumpet on `` Closing Time '' (
Instrumental ) Technical personnel ( edit ) Ed Caraeff -- photography Richie Moore -- additional engineering Cal Schenkel -- design , artwork Jerry Yester --
production , engineering Chart positions ( edit ) Chart ( 2000 ) Peak position Irish Albums Chart 29 Certifications ( edit ) Region Certification Certified
units / Sales United Kingdom ( BPI ) Gold 100,000 Summaries shipments figures based on certification alone References ( edit ) Jump up ^ `` Releases '' .
</item 2>

<item 3>
For What It's Worth
`` What is Country Rock ? '' . ThoughtCo . About.com . Retrieved March 12 , 2017 . Jump up ^ Jim DeRogatis ( 1996 ) . Kaleidoscope Eyes : Psychedelic Rock from
the ' 60s to the ' 90s . Carol Publishing Group . p. 51 . ISBN 978 - 0 - 8065 - 1788 - 9 . Retrieved 13 October 2015 . Jump up ^ Google Books archive Billboard
March 25 , 1967 retrieved 25 March 2017 Jump up ^ `` Acclaimed Music Top 3000 songs '' . Acclaimed Music . 27 May 2009 . ^ Jump up to : David Browne ( November
11 , 2016 ) . `` ' For What It 's Worth ' : Inside Buffalo Springfield 's Classic Protest Song '' . Rolling Stone . Jump up ^ Gilliland , John ( 1969 ) . ``
Show 34 -- Revolt of the Fat Angel : American musicians respond to the British invaders .
</item 3>

<item 4>
Let My Love Open the Door
Songfacts.com . Jump up ^ `` Pete Townshend UK charts '' . Official Charts . Jump up ^ `` Pete Townshend US charts '' . Billboard . Jump up ^ Joel Whitburn 's
Top Pop Singles 1955 - 1990 - ISBN 0 - 89820 - 089 - X Jump up ^ Collectionscanada.gc.ca Archived 2016 - 04 - 25 at the Wayback Machine . Jump up ^
Musicoutfitters.com Jump up ^ `` The Covers EP : Various artists : MP3 Downloads '' . Amazon.com . Retrieved 2016 - 10 - 16 . Jump up ^ `` Help us open the door
to better cancer care # LetMyLoveOpenTheDoor '' . Firstcitizens.com . Retrieved 2016 - 10 - 16 . Jump up ^ `` Walmart TV Commercial , ' Christmas Like a Rock
Star ' Song by Pete Townshend '' . Ispot.tv . Retrieved December 6 , 2017 . Jump up ^ `` Leathal Weapon ( TV series ) `` Fools Rush '' Soundtracks ( 2017 ) `` .
</item 4>

<item 5>
For What It's Worth
Entertainment.time.com . Retrieved 2016 - 09 - 30 . Jump up ^ `` Oui 3 '' . Blair Booth Music . Retrieved 6 April 2016 . Jump up ^ `` Oui 3 '' . Official Charts
. Retrieved 6 April 2016 . Jump up ^ `` EXCLUSIVE ! Haley Reinhart Covering The ' 60s Classic For What It 's Worth Is As Chilling As Our Current Political
Climate ! Listen ! '' . PerezHilton . August 10 , 2017 . Retrieved August 10 , 2017 . External links ( edit ) Reasontorock analysis of song Lyrics of this song
at MetroLyrics Buffalo Springfield Richie Furay Dewey Martin Jim Messina Stephen Stills Neil Young Bruce Palmer Studio albums Buffalo Springfield Buffalo
Springfield Again Last Time Around Compilations and box sets Retrospective : The Best of Buffalo Springfield Buffalo Springfield ( compilation album ) Buffalo
Springfield ( box set ) Songs `` For What It 's Worth '' `` Sit Down , I Think I Love You '' `` Mr.
</item 5>

<item 6>
Closing Time (album)
The songs on Closing Time are often noted for their lyrical content and vary in form . `` Ol ' ' 55 '' narrates the story of a man riding `` lickety splitly ''
in a car and is often seen as a song about escapism , `` like his near - contemporary Bruce Springsteen . '' This theme is also present in `` Old Shoes '' which
narrates another story about `` a footloose young stud hitting the road and semi-sneering '' , particularly in lyrics such as `` your tears can not bind me
anymore '' and `` my heart was not born to be tamed . '' Other lyrics on the album are described as melancholic , particularly `` Lonely '' , `` I Hope That I
Do n't Fall in Love with You '' and `` Grapefruit Moon '' , which are `` both self - conscious and lacklustre . '' The intro to `` Midnight Lullaby '' borrows
lyrics from the English nursery rhyme `` Sing a Song of Sixpence '' . This form of song - writing became a lifelong habit of Waits following the writing of ``
Midnight Lullaby '' , in which he `` assembled lyrics from fragments of oral tradition . ''
</item 6>

<item 7>
For What It's Worth
For What It 's Worth - wikipedia For What It 's Worth Jump to : navigation , search This article is about the Stephen Stills song . For other uses , see For
What It 's Worth ( disambiguation ) . `` What 's that sound ? '' redirects here . For the album , see What 's That Sound ? `` For What It 's Worth '' Single by
Buffalo Springfield from the album Buffalo Springfield B - side `` Do I Have to Come Right Out and Say It ? ''
</item 7>

<item 8>
What Ifs
The song was released with the album through RCA Nashville and was serviced to radio as the third single on February 6 , 2017 . Contents ( hide ) 1 Background 2
Composition 3 Commercial performance 4 Music video 5 Chart performance 5.1 Weekly charts 5.2 Year - end charts 6 Certifications and sales 7 References
Background ( edit ) The song was written by Brown with Matthew McGinn and Jordan Schmidt . The song is about a couple where the question `` what if '' is asked
repeatedly about what would happen in their relationship . It features a duet with Lauren Alaina , a school friend from Lakeview - Fort Oglethorpe High School
when he was living in Fort Oglethorpe , Georgia . According to Brown , the song was initially not intended to be a duet , he said : `` Once we decided on adding
a woman to `` What Ifs , '' I knew that I wanted Lauren to sing it , and , of course , she killed it . ''
</item 8>

<item 9>
Closing Time (album)
Critical reaction to Closing Time was positive . The album 's only single , `` Ol ' ' 55 '' , attracted attention due to a cover version by Waits 's more
popular label mates , the Eagles . Other songs from the album were covered by Tim Buckley and Bette Midler . The album was certified Gold in the UK and has
gained a contemporary cult following among rock fans . Since its release , the album has been reissued on LP in 1976 , on CD in 1992 and 1999 , and 180 gram LP
in 2010 . Contents ( hide ) 1 Background 2 Recording 3 Composition 4 Packaging 5 Release 6 Reception 7 Influence 8 Track listing 9 Personnel 9.1 Musicians 9.2
Guest musicians 9.3 Technical personnel 10 Chart positions 11 Certifications 11.1 References 11.2 Bibliography Background ( edit ) The Troubadour was the site
of many early Waits performances Tom Waits began his musical career in 1970 , performing every Monday night at The Troubadour , a venue in Los Angeles ,
California .
</item 9>

<item 10>
For What It's Worth
Released January 1967 Format 7 - inch single Recorded December 5 , 1966 Studio Columbia Studios , Hollywood Genre Folk rock country rock psychedelic rock Length
2 : 37 Label Atco Songwriter ( s ) Stephen Stills Producer ( s ) Charles Greene Brian Stone Buffalo Springfield singles chronology `` Burned '' ( 1966 ) `` For
What It 's Worth '' ( 1967 ) `` Bluebird '' ( 1967 ) `` Burned '' ( 1966 ) `` For What It 's Worth '' ( 1967 ) `` Bluebird '' ( 1967 ) `` For What It 's Worth (
Stop , Hey What 's That Sound ) '' ( often referred to as simply `` For What It 's Worth '' ) is a song written by Stephen Stills . It was performed by Buffalo
Springfield , recorded on December 5 , 1966 , and released as a single on Atco Records in January 1967 . The single peaked at number seven on the Billboard Hot
100 chart .
</item 10>
=== END ITEM id=e060aa54aef17fde ===

=== ITEM id=e7a8e2db59002f6a ===
Gold sentences and facts:
Sentence 0: The Ring of Fire is a direct result of plate tectonics : the movement and collisions of lithospheric plates . The eastern section of the ring is the result of the Nazca Plate and the Cocos Plate being subducted beneath the westward - moving South American Plate . The Cocos Plate is being subducted beneath the Caribbean Plate , in Central America . A portion of the Pacific Plate and the small Juan de Fuca Plate are being subducted beneath the North American Plate . Along the northern portion , the northwestward - moving Pacific plate is being subducted beneath the Aleutian Islands arc . Farther west , the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan . The southern portion is more complex , with a number of smaller tectonic plates in collision with the Pacific plate from the Mariana Islands , the Philippines , Bougainville , Tonga , and New Zealand ; this portion excludes Australia , since it lies in the center of its tectonic plate . Indonesia lies between the Ring of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt along the south and west from Sumatra , Java , Bali , Flores , and Timor . The famous and very active San Andreas Fault zone of California is a transform fault which offsets a portion of the East Pacific Rise under southwestern United States and Mexico . The motion of the fault generates numerous small earthquakes , at multiple times a day , most of which are too small to be felt . The active Queen Charlotte Fault on the west coast of the Haida Gwaii , British Columbia , has generated three large earthquakes during the 20th century : a magnitude 7 event in 1929 ; a magnitude 8.1 in 1949 ( Canada 's largest recorded earthquake ) ; and a magnitude 7.4 in 1970 .
  Fact 0.0: The Ring of Fire is a direct result of plate tectonics.
  Fact 0.1: Plate tectonics involves the movement and collisions of lithospheric plates.
  Fact 0.2: The eastern section of the Ring of Fire is the result of the Nazca Plate being subducted beneath the westward-moving South American Plate.
  Fact 0.3: The eastern section of the Ring of Fire is the result of the Cocos Plate being subducted beneath the westward-moving South American Plate.
  Fact 0.4: The Cocos Plate is being subducted beneath the Caribbean Plate in Central America.
  Fact 0.5: A portion of the Pacific Plate is being subducted beneath the North American Plate.
  Fact 0.6: The small Juan de Fuca Plate is being subducted beneath the North American Plate.
  Fact 0.7: Along the northern portion, the northwestward-moving Pacific plate is being subducted beneath the Aleutian Islands arc.
  Fact 0.8: Farther west, the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan.
  Fact 0.9: The southern portion of the Ring of Fire is more complex.
  Fact 0.10: In the southern portion, a number of smaller tectonic plates are in collision with the Pacific plate from the Mariana Islands, the Philippines, Bougainville, Tonga, and New Zealand.
  Fact 0.11: The southern portion of the Ring of Fire excludes Australia.
  Fact 0.12: Australia is excluded from the southern portion since it lies in the center of its tectonic plate.
  Fact 0.13: Indonesia lies between the Ring of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt.
  Fact 0.14: The Alpide belt runs along the south and west of Indonesia from Sumatra, Java, Bali, Flores, and Timor.
  Fact 0.15: The San Andreas Fault zone of California is famous and very active.
  Fact 0.16: The San Andreas Fault zone of California is a transform fault.
  Fact 0.17: The San Andreas Fault offsets a portion of the East Pacific Rise under southwestern United States and Mexico.
  Fact 0.18: The motion of the San Andreas Fault generates numerous small earthquakes, at multiple times a day.
  Fact 0.19: Most of the small earthquakes generated by the San Andreas Fault are too small to be felt.
  Fact 0.20: The Queen Charlotte Fault is active.
  Fact 0.21: The Queen Charlotte Fault is on the west coast of the Haida Gwaii, British Columbia.
  Fact 0.22: The Queen Charlotte Fault has generated three large earthquakes during the 20th century.
  Fact 0.23: One of the large Queen Charlotte Fault earthquakes was a magnitude 7 event in 1929.
  Fact 0.24: One of the large Queen Charlotte Fault earthquakes was a magnitude 8.1 event in 1949, which is Canada's largest recorded earthquake.
  Fact 0.25: One of the large Queen Charlotte Fault earthquakes was a magnitude 7.4 event in 1970.

Retrieved items:
<item 1>
Ring of Fire
The southern portion is more complex , with a number of smaller tectonic plates in collision with the Pacific plate from the Mariana Islands , the Philippines ,
Bougainville , Tonga , and New Zealand ; this portion excludes Australia , since it lies in the center of its tectonic plate . Indonesia lies between the Ring
of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt along the south and west from Sumatra , Java , Bali , Flores ,
and Timor . The famous and very active San Andreas Fault zone of California is a transform fault which offsets a portion of the East Pacific Rise under
southwestern United States and Mexico . The motion of the fault generates numerous small earthquakes , at multiple times a day , most of which are too small to
be felt . The active Queen Charlotte Fault on the west coast of the Haida Gwaii , British Columbia , has generated three large earthquakes during the 20th
century : a magnitude 7 event in 1929 ; a magnitude 8.1 in 1949 ( Canada 's largest recorded earthquake ) ; and a magnitude 7.4 in 1970 .
</item 1>

<item 2>
Ring of Fire
Ring of Fire - wikipedia Ring of Fire Jump to : navigation , search For other uses , see Ring of Fire ( disambiguation ) . The Pacific Ring of Fire Tectonic
plates of the world Great ( magnitude 8.0 + ) earthquakes 1900 -- 2017 , with the majority positioned along the Ring of Fire The Ring of Fire is a major area in
the basin of the Pacific Ocean where a large number of earthquakes and volcanic eruptions occur . In a 40,000 km ( 25,000 mi ) horseshoe shape , it is
associated with a nearly continuous series of oceanic trenches , volcanic arcs , and volcanic belts and plate movements . It has 452 volcanoes ( more than 75 %
of the world 's active and dormant volcanoes ) . The Ring of Fire is sometimes called the circum - Pacific belt . About 90 % of the world 's earthquakes and 81
% of the world 's largest earthquakes occur along the Ring of Fire .
</item 2>

<item 3>
Ring of Fire
Sea Scotia Sea Somov Sea Weddell Sea Landlocked seas Aral Sea Caspian Sea Dead Sea Salton Sea Book Related subjects and articles Countries of the Pacific Ring
of Fire Belize Bolivia Brazil Canada Colombia Chile Costa Rica Ecuador East Timor El Salvador Micronesia Fiji Guatemala Honduras Indonesia Japan Kiribati Mexico
New Zealand Nicaragua Palau Papua New Guinea Panama Peru Philippines Russia Samoa Solomon Islands Tonga Tuvalu United States Oceanic Trenches of the Pacific
Ring of Fire Aleutian Trench Bougainville Trench Kermadec Trench Izu Bonin Trench Japan Trench Kurile Trench Mariana Trench Middle America Trench Peru -- Chile
Trench Philippine Trench Ryukyu Trench Tonga Trench Yap Trench Tectonic Plates of the Pacific Ring of Fire Antarctic Plate Australian Plate Caribbean Plate
Cocos Plate Eurasian Plate Explorer Plate Gorda Plate Juan de Fuca Plate Nazca Plate North American Plate Pacific Plate Philippine Sea Plate Rivera Plate South
American Plate Volcanoes of the Pacific Ring of Fire Baker Bulusan Cold Bay Concepción Fuego Fuji Galeras Hood Mount Erebus Krakatoa Mayon Cascade Volcanoes
Merapi Momotombo Novarupta Parícutin Pico de
</item 3>

<item 4>
Ring of Fire
Cascade Volcanoes Merapi Momotombo Novarupta Parícutin Pico de Orizaba Pinatubo Popocatépetl Shasta Rainier Ruapehu Nevado del Ruiz Saint Helens Tambora
Taranaki Tungurahua Usu Hoodoo Edziza Tseax The Volcano Meager Garibaldi Cayley Silverthrone Volcano Mountain Other Notable Geographic Features of the Pacific
Ring of Fire Alaska Aleutian Islands American cordillera Andes Antarctica Bali Borneo Bougainville British Columbia California Cascade Range Challenger Deep
Coast Mountains Fais Island Flores Guam Honshū Insular Mountains Java Kamchatka Kurile Islands Luzon Mariana Islands Melanesia Micronesia Mindanao New Guinea
North Island ( NZL ) Northern Mariana Islands Oregon Pacific Coast Range Pacific Ocean Polynesia Rocky Mountains San Andreas Fault Queen Charlotte Fault Sierra
Madre ( PHL ) Sierra Madre ( USA ) Sierra Madre del Sur Sierra Madre Occidental Sierra Madre Oriental South Island ( NZL ) Sulawesi Timor Washington Yap Yukon
Territory Retrieved from `` https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Graph query row:
entity: Ring of Fire
relationship: RELATED_TO
region: Geography of Oceania
</item 5>

<item 6>
Ring of Fire
Islands Nauru Northern Mariana Islands Wake Island Polynesia Easter Island Hawaiian Islands Cook Islands French Polynesia Austral Islands Gambier Islands
Marquesas Islands Society Islands Tuamotu Kermadec Islands Mangareva Islands Samoa Tokelau Tonga Tuvalu Ring of Fire Regions of South America East Amazon basin
Atlantic Forest Caatinga Cerrado North Caribbean South America West Indies Los Llanos The Guianas Amazon basin Amazon rainforest Gulf of Paria Paria Peninsula
Paraguaná Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran Chaco Chiquitano dry forests Valdes Peninsula West Andes Tropical Andes
Wet Andes Dry Andes Pariacaca mountain range Altiplano Atacama Desert Latin Hispanic American Cordillera Ring of Fire LAC Polar regions Antarctic Antarctic
Peninsula East Antarctica West Antarctica Eklund Islands Ecozone Extreme points Islands Arctic Arctic Alaska British Arctic Territories Canadian Arctic
Archipelago Finnmark Greenland Northern Canada Northwest Territories Nunavik Nunavut Russian Arctic Sakha Sápmi Yukon North American Arctic Earth 's oceans and
seas Arctic Ocean Amundsen Gulf Barents Sea Beaufort Sea Chukchi Sea East
</item 6>

<item 7>
Ring of Fire
Trans - Mexican Volcanic Belt Central America Volcanic Arc Andes North Volcanic Zone Central Volcanic Zone South Volcanic Zone Austral Volcanic Zone Scotia Arc
South Sandwich Islands Antarctica Antarctic Peninsula Victoria Land See also ( edit ) Earthquakes portal Volcanoes portal 2004 Indian Ocean earthquake and
tsunami 2010 Chile earthquake 2011 Tōhoku earthquake and tsunami Andesite line Geology of the Pacific Northwest Katsuhiko Ishibashi Pacific Rim References (
edit ) Jump up ^ `` Ring of Fire -- Pacific Ring of Fire '' . Geography.about.com. 2010 - 06 - 14 . Retrieved 2010 - 11 - 01 . Jump up ^ `` Ring of Fire '' .
USGS. 2012 - 07 - 24 . Retrieved 2013 - 06 - 13 . Jump up ^ `` Where do earthquakes occur ? '' . USGS. 2013 - 05 - 13 . Archived from the original on 2014 - 08
- 05 . Retrieved 2013 - 06 - 13 .
</item 7>

<item 8>
Ring of Fire
When the `` locked '' zone stores energy for an earthquake , the `` transition '' zone , although somewhat plastic , can rupture . Thermal and deformation
studies indicate that the locked zone is fully locked for 60 km ( about 40 mi ) down - dip from the deformation front . Further down - dip , a transition from
fully locked to aseismic sliding occurs . American Cascade Range volcano eruptions in the last 4000 years Unlike most subduction zones worldwide , no oceanic
trench is present along the continental margin in Cascadia . Instead , terranes and the accretionary wedge have been lifted up to form a series of coast ranges
and exotic mountains . A high rate of sedimentation from the outflow of the three major rivers ( Fraser River , Columbia River , and Klamath River ) which cross
the Cascade Range contributes to further obscuring the presence of a trench . However , in common with most other subduction zones , the outer margin is slowly
being compressed , similar to a giant spring .
</item 8>

<item 9>
Ring of Fire
The next most seismically active region ( 5 -- 6 % of earthquakes and 17 % of the world 's largest earthquakes ) is the Alpide belt , which extends from Java to
the northern Atlantic Ocean via the Himalayas and southern Europe . All but three of the world 's 25 largest volcanic eruptions of the last 11,700 years
occurred at volcanoes in the Ring of Fire . The Ring of Fire is a direct result of plate tectonics : the movement and collisions of lithospheric plates . The
eastern section of the ring is the result of the Nazca Plate and the Cocos Plate being subducted beneath the westward - moving South American Plate . The Cocos
Plate is being subducted beneath the Caribbean Plate , in Central America . A portion of the Pacific Plate and the small Juan de Fuca Plate are being subducted
beneath the North American Plate . Along the northern portion , the northwestward - moving Pacific plate is being subducted beneath the Aleutian Islands arc .
Farther west , the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan .
</item 9>

<item 10>
Ring of Fire
Land areas ( edit ) Papua New Guinea and tectonic plates : Pacific Plate , Australian Plate , Caroline Plate , Banda Sea Plate ( as `` Mer de Banda '' ) ,
Woodlark Plate , Bird 's Head Plate , Maoke Plate , Solomon Sea Plate , North Bismarck Plate , South Bismarck Plate and Manus Plate ( in French ) Puysegur
Trench and Macquarie Ridge Taupo Volcanic Zone Kermadec Islands Tonga Islands Fiji Islands Bougainville and Solomon Islands New Hebrides Arc Bismarck Volcanic
Arc , junction Tanimbar and Kai Islands Lesser Sunda Islands Sunda Arc Andaman and Nicobar Islands Izu - Bonin - Mariana Arc Mariana Islands Bonin Islands Izu
Islands Philippine Mobile Belt Taiwan Ryukyu Islands Japan , Boso Triple Junction Kuril Islands Kamchatka Peninsula American Cordillera North American
Cordillera Aleutian Arc Aleutian Islands Aleutian Range Eastern Alaska , U.S. Northern Cordilleran Volcanic Province Cascade Volcanic Arc and Rio Grande rift
Sierra Nevada ( U.S. )
</item 10>
=== END ITEM id=e7a8e2db59002f6a ===

=== ITEM id=cf5a0f65f41d51ba ===
Gold sentences and facts:
Sentence 0: `` The Gift of the Magi '' is a short story , written by O. Henry ( a pen name for William Sydney Porter ) , about a young husband and wife and how they deal with the challenge of buying secret Christmas gifts for each other with very little money . As a sentimental story with a moral lesson about gift - giving , it has been a popular one for adaptation , especially for presentation at Christmas time . The plot and its twist ending are well - known , and the ending is generally considered an example of comic irony . It was allegedly written at Pete 's Tavern on Irving Place in New York City .
  Fact 0.0: The Gift of the Magi is a short story.
  Fact 0.1: The Gift of the Magi was written by O. Henry.
  Fact 0.2: O. Henry is a pen name for William Sydney Porter.
  Fact 0.3: The Gift of the Magi is about a young husband and wife.
  Fact 0.4: The husband and wife deal with the challenge of buying secret Christmas gifts for each other with very little money.
  Fact 0.5: The Gift of the Magi is a sentimental story with a moral lesson about gift-giving.
  Fact 0.6: The Gift of the Magi has been a popular story for adaptation.
  Fact 0.7: The Gift of the Magi has been especially popular for presentation at Christmas time.
  Fact 0.8: The plot of The Gift of the Magi is well-known.
  Fact 0.9: The twist ending of The Gift of the Magi is well-known.
  Fact 0.10: The ending of The Gift of the Magi is generally considered an example of comic irony.
  Fact 0.11: The Gift of the Magi was allegedly written at Pete's Tavern on Irving Place in New York City.

Retrieved items:
<item 1>
The Gift of the Magi
Henry The Gift of the Magi public domain audiobook at LibriVox O'Henry's `` The Gift of the Magi '' : Common Core Exemplar from NEH 's EDSITEment `` The Gift of
the Magi '' study guide , teacher resources , themes , quotes ( hide ) O. Henry Short stories `` The Duplicity of Hargraves '' ( 1902 ) `` A Retrieved
Reformation '' ( 1903 ) `` The Cop and the Anthem '' ( 1904 ) `` The Gift of the Magi '' ( 1905 ) `` Conscience in Art '' ( 1907 ) `` The Last Leaf '' ( 1907 )
`` The Third Ingredient '' ( 1908 ) `` The Ransom of Red Chief '' ( 1910 ) `` Makes the Whole World Kin '' ( 1911 ) Novels Cabbages and Kings ( 1904 ) Story
collections The Four Million ( 1906 ) Roads of Destiny ( 1909 ) Whirligigs ( 1910 ) Waifs and Strays ( 1917 ) Related O. Henry Award William Sidney Porter House
O. Henry House and Museum O. Henry Hall O.
</item 1>

<item 2>
The Gift of the Magi
Henry Award William Sidney Porter House O. Henry House and Museum O. Henry Hall O. Henry 's Full House ( 1952 ) Strictly Business ( 1962 ) The Trust That Went
Bust Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Gift_of_the_Magi&oldid=816945958 '' Categories : 1906 short stories Christmas short
stories Irony Short stories by O. Henry Short stories adapted into films Works originally published in the New York World Hidden categories : Pages using
deprecated image syntax Articles with LibriVox links Talk Contents About Wikipedia Wikisource Deutsch فارسی 한국어 Հայերեն 日本 語 ਪੰਜਾਬੀ Português Русский Simple
English Slovenščina Tagalog Українська Tiếng Việt 中文 Edit links This page was last edited on 24 December 2017 , at 22 : 11 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 2>

<item 3>
The Gift of the Magi
The Gift of the Magi - wikipedia The Gift of the Magi Jump to : navigation , search This article is about the short story . For the Magi that visited baby Jesus
, see Biblical Magi . For other uses , see The Gift of the Magi ( disambiguation ) . `` The Gift of the Magi '' Author O. Henry Country United states Language
English Genre ( s ) Short story Published in The Four Million Publication type Anthology Publication date December 10 , 1905 ( newspaper ) ; April 10 , 1906 (
book ) `` The Gift of the Magi '' is a short story , written by O. Henry ( a pen name for William Sydney Porter ) , about a young husband and wife and how they
deal with the challenge of buying secret Christmas gifts for each other with very little money . As a sentimental story with a moral lesson about gift - giving
, it has been a popular one for adaptation , especially for presentation at Christmas time . The plot and its twist ending are well - known , and the ending is
generally considered an example of comic irony . It was allegedly written at Pete 's Tavern on Irving Place in New York City .
</item 3>

<item 4>
The Gift of the Magi
LiteraryTraveler.com . Jump up ^ Не буду гангстером , дорогая Jump up ^ Mickey 's Once Upon A Christmas on IMDb Jump up ^ O. Henry on IMDb Jump up ^ Raincoat on
IMDb Jump up ^ The Gift of the Magi ( 2014 ) - IMDb Jump up ^ Darovete Na Vlahvite ( 2013 ) - IMDb Jump up ^ St. Germain , Mark ; Courts , Randy ( 1984 ) . The
Gifts of the Magi . New York : Dramatists Play Service , Inc . ISBN 978 - 0 - 8222 - 1461 - 8 Jump up ^ Watch A Trump Christmas from Saturday Night Live on
NBC.com , retrieved 2017 - 08 - 03 External links ( edit ) Wikisource has original text related to this article : The Gift of the Magi `` The Gift of the Magi
'' by O.
</item 4>

<item 5>
The Gift of the Magi
Unusually late , Jim walks in and immediately stops short at the sight of Della , who had previously prayed that she was still pretty to Jim . Della then admits
to Jim that she sold her hair to buy him his present . Jim gives Della her present -- an assortment of combs , useless now that her hair is short . Della then
shows Jim the chain she bought for him , to which Jim says he sold his watch to get the money to buy her combs . Although Jim and Della are now left with gifts
that neither one can use , they realize how far they are willing to go to show their love for each other , and how priceless their love really is . The story
ends with the narrator comparing the pair 's mutually sacrificial gifts of love with those of the Biblical Magi .
</item 5>

<item 6>
The Gift of the Magi
The special premiered on HBO on December 17 , 1978 . The special later aired on ABC in 1980 and on Nickelodeon in the 1990s . The special features several
original songs written by songwriter Paul Williams . The sketch comedy show Studio C contains a skit called `` Gift of the Magi '' in which Jim and Della argue
over their gift - giving . The television series My Little Pony : Friendship is Magic has the episode `` The Gift of the Maud Pie '' in which Pinkie Pie gives
Maud a perfect gift , costing her the confetti - using Party Cannon . Maud 's gift in return was confetti for the cannon of which Pinkie Pie had sold . Radio
drama series Adventures in Odyssey features a comedic adaptation of this story in its episode `` Gifts for Madge & Guy '' The animated sketch comedy series
Robot Chicken features a parody of the story in its fourth season , except that Jim does not sell his watch and instead buys Della lingerie , much to her
consternation . Joni Mitchell wrote and performed , but never recorded , a song based on the story .
</item 6>

<item 7>
The Gift of the Magi
It was allegedly written at Pete 's Tavern on Irving Place in New York City . The story was initially published in The New York Sunday World under the title ``
Gifts of the Magi '' on December 10 , 1905 . It was first published in book form in the O. Henry Anthology The Four Million in April 1906 . Contents ( hide ) 1
Summary 2 Adaptations 3 References 4 External links Summary ( edit ) Mr. James Dillingham Young ( `` Jim '' ) and his wife , Della , are a couple living in a
modest apartment . On Christmas Eve , with only $1.87 in hand , and desperate to find a gift for Jim , Della sells her hair for $20 to a nearby hairdresser
named Madame Sofronie , and eventually finds a platinum pocket watch fob chain for Jim 's watch for $21 . Satisfied with the perfect gift for Jim , Della runs
home and begins to prepare pork chops for dinner . At 7 o'clock , Della sits at a table near the door , waiting for Jim to come home .
</item 7>

<item 8>
The Gift of the Magi
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 8>

<item 9>
The Gift of the Magi
The Greek film directed by Ismene Daskarolis ( 2014 ) places it in the economical crisis of Greece today . The Mexican film Nosotros los pobres includes this
tale as a small sub-plot . There is also a Bulgarian short film adaptation known as `` Darovete na vlahvite '' ( 2013 ) directed by Ivan Abadjiev . An off -
Broadway musical version entitled The Gifts of the Magi premiered at Lamb 's Theatre in New York City in 1984 . Written by Mark St. Germain and Randy Courts ,
the play is regularly produced in schools and regional theaters . In 1955 , in The Honeymooners episode 13 of season 1 , `` ' Twas the Night Before Christmas ''
involves Ralph looking for the perfect present for Alice for Christmas . Without money , he pawns his prized bowling ball to buy her an expensive gift , only to
find out in the end that she bought him a custom bowling ball bag .
</item 9>

<item 10>
The Gift of the Magi
Adaptations ( edit ) The story has been adapted to films , The Sacrifice ( 1909 ) , Love 's Surprises Are Futile ( 1916 ) , The Gift of the Magi ( 1917 ) , a
segment of O. Henry 's Full House ( 1952 ) , The Gift of Love ( 1978 ) , The Gift of the Magi ( 1958 ) , Dary magów ( Poland , 1972 ) , Christmas Eve on Sesame
Street ( 1978 ) , I 'll not be a gangster , love ( Не буду гангстером , дорогая / Nebūsiu gangsteriu , brangioji , USSR , 1978 ) , Mickey 's Once Upon a
Christmas ( 1999 ) , The Gift of the Magi ( 2004 ) and the short film for the Irish band The Script in 2010 called For the First Time . Love , another French
movie , based some of its scenes on this story . Raincoat ( 2004 ) , a Hindi film directed by Rituparno Ghosh is an adaptation of the story .
</item 10>
=== END ITEM id=cf5a0f65f41d51ba ===

=== ITEM id=1e2b9a5865a9b1cb ===
Gold sentences and facts:
Sentence 0: Economy of India Mumbai , Maharashtra the financial center of India Currency Indian rupee ( INR ) ₹ 1 = 100 Paise Fiscal year 1 April -- 31 March Trade organisations WTO , WCO , WFTU , G - 20 and others Statistics GDP $2.848 trillion ( nominal ; 2018 est ) $10.385 trillion ( PPP ; 2018 est ) GDP rank 6th ( nominal ) 3rd ( PPP ) GDP growth 7.2 % ( Q3 , 2017 - 18 ) ( MOSPI ) GDP per capita $2,134 ( nominal ; 2018 est ) $7,783 ( PPP ; 2018 est ) GDP per capita rank 141st ( nominal ) 123rd ( PPP ) GDP by sector Agriculture : 17.32 % Industry : 29.02 % Services : 53.66 % ( 2016 est . ) Inflation ( CPI ) 4.28 % ( March 2018 ) Base borrowing rate 6.25 % ( as on 16 Mar. 2018 ) Population below poverty line 21.2 % 58 % live less than $3.10 / day ( 2011 ; World bank estimate ) Gini coefficient 33.9 ( 2013 ) Human Development Index 0.624 ( 2015 ) medium ( 131st ) Labour force 520.2 million ( 2017 ) Labour force by occupation Agriculture : 47 % Industry : 22 % Services : 31 % ( FY 2014 est . ) Unemployment 8.8 % ( 2017 ) Average net salary ₹ 82,269 / $1,284 annually ( 2016 - 17 ) Main industries Textiles chemicals food processing steel cement mining petroleum machinery software pharmaceuticals transportation equipment Ease - of - doing - business rank 100 ( 2017 ) External Exports $275.8 billion ( 2017 ) Export goods Agricultural products 13.2 % Fuels and mining products 15.7 % Manufacturers 68.4 % Others 2.7 % Main export partners European Union 17.6 % United States 16.1 % United Arab Emirates 11.5 % Hong Kong 5.1 % China 3.4 % Other 46.3 % Imports $384.3 billion ( 2017 ) Import goods Agricultural products 7.1 % Fuels and mining products 33.1 % Manufacturers 47.8 % Other 12 % Main import partners China 17 % European Union 11.3 % United States 5.7 % United Arab Emirates 5.4 % Saudi Arabia 5.2 % Other 55.5 % FDI stock Inward : $318.50 billion Outward : $144.13 billion ( 2016 ) Current account 0.7 % of GDP ( 2016 -- 17 ) Gross external debt $471.9 billion ( 31 March 2017 ) Net international investment position - $404.8 billion ( Sept. 2017 ) Public finances Public debt 67.7 % of GDP ( 2017 ) Revenues ₹ 35.41 trillion ( US $540 billion ) ( 2017 ) Expenses ₹ 46.25 trillion ( US $710 billion ) ( 2017 ) Economic aid $2.68 billion ( 2016 ) Credit rating Standard & Poor 's : BBB − ( Domestic ) BBB − ( Foreign ) BBB+ ( T&C Assessment ) Outlook : Stable Moody 's : Baa2 Outlook : Stable Fitch : BBB − Outlook : Stable Foreign reserves $426.082 billion ( 13 April 2018 ) ( 8th ) Main data source : CIA World Fact Book All values , unless otherwise stated , are in US dollars .
  Fact 0.0: Mumbai, Maharashtra is the financial center of India.
  Fact 0.1: The currency of India is the Indian rupee (INR).
  Fact 0.2: 1 rupee equals 100 paise.
  Fact 0.3: India's fiscal year runs from 1 April to 31 March.
  Fact 0.4: India's trade organisations include WTO, WCO, WFTU, G-20 and others.
  Fact 0.5: India's nominal GDP is $2.848 trillion (2018 estimate).
  Fact 0.6: India's PPP GDP is $10.385 trillion (2018 estimate).
  Fact 0.7: India's nominal GDP rank is 6th.
  Fact 0.8: India's PPP GDP rank is 3rd.
  Fact 0.9: India's GDP growth is 7.2% (Q3, 2017-18), per MOSPI.
  Fact 0.10: India's nominal GDP per capita is $2,134 (2018 estimate).
  Fact 0.11: India's PPP GDP per capita is $7,783 (2018 estimate).
  Fact 0.12: India's nominal GDP per capita rank is 141st.
  Fact 0.13: India's PPP GDP per capita rank is 123rd.
  Fact 0.14: Agriculture makes up 17.32% of India's GDP by sector (2016 estimate).
  Fact 0.15: Industry makes up 29.02% of India's GDP by sector (2016 estimate).
  Fact 0.16: Services make up 53.66% of India's GDP by sector (2016 estimate).
  Fact 0.17: India's CPI inflation is 4.28% (March 2018).
  Fact 0.18: India's base borrowing rate is 6.25% (as on 16 March 2018).
  Fact 0.19: 21.2% of India's population is below the poverty line.
  Fact 0.20: 58% of India's population lives on less than $3.10 per day (2011, World Bank estimate).
  Fact 0.21: India's Gini coefficient is 33.9 (2013).
  Fact 0.22: India's Human Development Index is 0.624 (2015), rated medium and ranked 131st.
  Fact 0.23: India's labour force is 520.2 million (2017).
  Fact 0.24: Agriculture accounts for 47% of India's labour force by occupation (FY 2014 estimate).
  Fact 0.25: Industry accounts for 22% of India's labour force by occupation (FY 2014 estimate).
  Fact 0.26: Services account for 31% of India's labour force by occupation (FY 2014 estimate).
  Fact 0.27: India's unemployment rate is 8.8% (2017).
  Fact 0.28: India's average net salary is ₹82,269 or $1,284 annually (2016-17).
  Fact 0.29: Main industries in India include textiles, chemicals, food processing, steel, cement, mining, petroleum, machinery, software, pharmaceuticals and transportation equipment.
  Fact 0.30: India's ease-of-doing-business rank is 100 (2017).
  Fact 0.31: India's exports are $275.8 billion (2017).
  Fact 0.32: Agricultural products make up 13.2% of India's export goods.
  Fact 0.33: Fuels and mining products make up 15.7% of India's export goods.
  Fact 0.34: Manufacturers make up 68.4% of India's export goods.
  Fact 0.35: Other export goods make up 2.7% of India's exports.
  Fact 0.36: The European Union is 17.6% of India's main export partners.
  Fact 0.37: The United States is 16.1% of India's main export partners.
  Fact 0.38: The United Arab Emirates is 11.5% of India's main export partners.
  Fact 0.39: Hong Kong is 5.1% of India's main export partners.
  Fact 0.40: China is 3.4% of India's main export partners.
  Fact 0.41: Other export partners make up 46.3% of India's exports.
  Fact 0.42: India's imports are $384.3 billion (2017).
  Fact 0.43: Agricultural products make up 7.1% of India's import goods.
  Fact 0.44: Fuels and mining products make up 33.1% of India's import goods.
  Fact 0.45: Manufacturers make up 47.8% of India's import goods.
  Fact 0.46: Other import goods make up 12% of India's imports.
  Fact 0.47: China is 17% of India's main import partners.
  Fact 0.48: The European Union is 11.3% of India's main import partners.
  Fact 0.49: The United States is 5.7% of India's main import partners.
  Fact 0.50: The United Arab Emirates is 5.4% of India's main import partners.
  Fact 0.51: Saudi Arabia is 5.2% of India's main import partners.
  Fact 0.52: Other import partners make up 55.5% of India's imports.
  Fact 0.53: India's inward FDI stock is $318.50 billion (2016).
  Fact 0.54: India's outward FDI stock is $144.13 billion (2016).
  Fact 0.55: India's current account is 0.7% of GDP (2016-17).
  Fact 0.56: India's gross external debt is $471.9 billion (31 March 2017).
  Fact 0.57: India's net international investment position is -$404.8 billion (Sept. 2017).
  Fact 0.58: India's public debt is 67.7% of GDP (2017).
  Fact 0.59: India's public revenues are ₹35.41 trillion, or US$540 billion (2017).
  Fact 0.60: India's public expenses are ₹46.25 trillion, or US$710 billion (2017).
  Fact 0.61: India's economic aid is $2.68 billion (2016).
  Fact 0.62: Standard & Poor's gives India a BBB- domestic credit rating.
  Fact 0.63: Standard & Poor's gives India a BBB- foreign credit rating.
  Fact 0.64: Standard & Poor's gives India a BBB+ T&C assessment.
  Fact 0.65: Standard & Poor's outlook for India is stable.
  Fact 0.66: Moody's gives India a Baa2 rating with a stable outlook.
  Fact 0.67: Fitch gives India a BBB- rating with a stable outlook.
  Fact 0.68: India's foreign reserves are $426.082 billion (13 April 2018), ranked 8th.
  Fact 0.69: The main data source for these figures is the CIA World Fact Book.
  Fact 0.70: All values, unless otherwise stated, are in US dollars.
Sentence 1: India has one of the fastest growing service sectors in the world with an annual growth rate above 9 % since 2001 , which contributed to 57 % of GDP in 2012 -- 13 . India has become a major exporter of IT services , Business Process Outsourcing ( BPO ) services , and software services with $154 billion revenue in FY 2017 . This is the fastest - growing part of the economy . The IT industry continues to be the largest private - sector employer in India . India is the third - largest start - up hub in the world with over 3,100 technology start - ups in 2014 -- 15 The agricultural sector is the largest employer in India 's economy but contributes to a declining share of its GDP ( 17 % in 2013 -- 14 ) . India ranks second worldwide in farm output . The industry ( manufacturing ) sector has held a steady share of its economic contribution ( 26 % of GDP in 2013 -- 14 ) . The Indian automobile industry is one of the largest in the world with an annual production of 21.48 million vehicles ( mostly two and three - wheelers ) in 2013 -- 14 . India had $600 billion worth of retail market in 2015 and one of world 's fastest growing e-commerce markets .
  Fact 1.0: India has one of the fastest growing service sectors in the world.
  Fact 1.1: India's service sector has grown at an annual rate above 9% since 2001.
  Fact 1.2: India's service sector contributed 57% of GDP in 2012-13.
  Fact 1.3: India has become a major exporter of IT services, Business Process Outsourcing (BPO) services and software services.
  Fact 1.4: India's IT, BPO and software services exports had $154 billion in revenue in FY 2017.
  Fact 1.5: The IT and services export part of the economy is the fastest-growing part of the economy.
  Fact 1.6: The IT industry is the largest private-sector employer in India.
  Fact 1.7: India is the third-largest start-up hub in the world.
  Fact 1.8: India had over 3,100 technology start-ups in 2014-15.
  Fact 1.9: The agricultural sector is the largest employer in India's economy.
  Fact 1.10: Agriculture's share of India's GDP is declining, at 17% in 2013-14.
  Fact 1.11: India ranks second worldwide in farm output.
  Fact 1.12: The industry (manufacturing) sector has held a steady share of India's economic contribution, at 26% of GDP in 2013-14.
  Fact 1.13: The Indian automobile industry is one of the largest in the world.
  Fact 1.14: The Indian automobile industry produced 21.48 million vehicles in 2013-14, mostly two- and three-wheelers.
  Fact 1.15: India had a $600 billion retail market in 2015.
  Fact 1.16: India has one of the world's fastest growing e-commerce markets.
Sentence 2: India ranks second worldwide in farm output . Agriculture and allied sectors like forestry , logging and fishing accounted for 17 % of the GDP and employed 49 % of its total workforce in 2014 . Agriculture accounted for 23 % of GDP , and employed 59 % of the country 's total workforce in 2016 . As the Indian economy has diversified and grown , agriculture 's contribution to GDP has steadily declined from 1951 to 2011 , yet it is still the country 's largest employment source and a significant piece of its overall socio - economic development . Crop - yield - per - unit - area of all crops has grown since 1950 , due to the special emphasis placed on agriculture in the five - year plans and steady improvements in irrigation , technology , application of modern agricultural practices and provision of agricultural credit and subsidies since the Green Revolution in India . However , international comparisons reveal the average yield in India is generally 30 % to 50 % of the highest average yield in the world . The states of Uttar Pradesh , Punjab , Haryana , Madhya Pradesh , Andhra Pradesh , Telangana , Bihar , West Bengal , Gujarat and Maharashtra are key contributors to Indian agriculture .
  Fact 2.0: India ranks second worldwide in farm output.
  Fact 2.1: Agriculture and allied sectors, such as forestry, logging and fishing, accounted for 17% of India's GDP in 2014.
  Fact 2.2: Agriculture and allied sectors employed 49% of India's total workforce in 2014.
  Fact 2.3: Agriculture accounted for 23% of India's GDP in 2016.
  Fact 2.4: Agriculture employed 59% of India's total workforce in 2016.
  Fact 2.5: Agriculture's contribution to India's GDP has steadily declined from 1951 to 2011.
  Fact 2.6: Agriculture is still the country's largest employment source.
  Fact 2.7: Agriculture is a significant piece of India's overall socio-economic development.
  Fact 2.8: Crop yield per unit area of all crops in India has grown since 1950.
  Fact 2.9: Crop yield growth in India is due to the special emphasis on agriculture in the five-year plans.
  Fact 2.10: Crop yield growth in India is due to steady improvements in irrigation and technology.
  Fact 2.11: Crop yield growth in India is due to application of modern agricultural practices.
  Fact 2.12: Crop yield growth in India is due to provision of agricultural credit and subsidies since the Green Revolution.
  Fact 2.13: International comparisons reveal that the average yield in India is generally 30% to 50% of the highest average yield in the world.
  Fact 2.14: Uttar Pradesh, Punjab, Haryana, Madhya Pradesh, Andhra Pradesh, Telangana, Bihar, West Bengal, Gujarat and Maharashtra are key contributors to Indian agriculture.

Retrieved items:
<item 1>
Agriculture in India
Illiteracy , general socio - economic backwardness , slow progress in implementing land reforms and inadequate or inefficient finance and marketing services for
farm produce . Inconsistent government policy . Agricultural subsidies and taxes are often changed without notice for short term political ends . Irrigation
facilities are inadequate , as revealed by the fact that only 52.6 % of the land was irrigated in 2003 -- 04 , which result in farmers still being dependent on
rainfall , specifically the monsoon season . A good monsoon results in a robust growth for the economy , while a poor monsoon leads to a sluggish growth . Farm
credit is regulated by NABARD , which is the statutory apex agent for rural development in the subcontinent . At the same time , over-pumping made possible by
subsidised electric power is leading to an alarming drop in aquifer levels . A third of all food that is produced rots due to inefficient supply chains and the
use of the `` Walmart model '' to improve efficiency is blocked by laws against foreign investment in the retail sector .
</item 1>

<item 2>
Agriculture in India
For food staples , the annual growth rate in production during the six - year segments 1970 - 76 , 1976 -- 82 , 1982 -- 88 , 1988 -- 1994 , 1994 - 2000 were
found to be respectively 2.5 , 2.5 , 3.0 , 2.6 , and 1.8 % per annum . Corresponding analyses for the index of total agricultural production show a similar
pattern , with the growth rate for 1994 - 2000 attaining only 1.5 % per annum . Infrastructure ( edit ) India has very poor rural roads affecting timely supply
of inputs and timely transfer of outputs from Indian farms . Irrigation systems are inadequate , leading to crop failures in some parts of the country because
of lack of water . In other areas regional floods , poor seed quality and inefficient farming practices , lack of cold storage and harvest spoilage cause over
30 % of farmer 's produce going to waste , lack of organised retail and competing buyers thereby limiting Indian farmer 's ability to sell the surplus and
commercial crops .
</item 2>

<item 3>
Agriculture in India
The economic contribution of agriculture to India 's GDP is steadily declining with the country 's broad - based economic growth . Still , agriculture is
demographically the broadest economic sector and plays a significant role in the overall socio - economic fabric of India . India exported $38 billion worth of
agricultural products in 2013 , making it the seventh largest agricultural exporter worldwide and the sixth largest net exporter . Most of its agriculture
exports serve developing and least developed nations . Indian agricultural / horticultural and processed foods are exported to more than 120 countries ,
primarily in the Middle East , Southeast Asia , SAARC countries , the EU and the United States .
</item 3>

<item 4>
Economy of India
Agriculture and allied sectors like forestry , logging and fishing accounted for 17 % of the GDP and employed 49 % of its total workforce in 2014 . Agriculture
accounted for 23 % of GDP , and employed 59 % of the country 's total workforce in 2016 . As the Indian economy has diversified and grown , agriculture 's
contribution to GDP has steadily declined from 1951 to 2011 , yet it is still the country 's largest employment source and a significant piece of its overall
socio - economic development . Crop - yield - per - unit - area of all crops has grown since 1950 , due to the special emphasis placed on agriculture in the
five - year plans and steady improvements in irrigation , technology , application of modern agricultural practices and provision of agricultural credit and
subsidies since the Green Revolution in India . However , international comparisons reveal the average yield in India is generally 30 % to 50 % of the highest
average yield in the world . The states of Uttar Pradesh , Punjab , Haryana , Madhya Pradesh , Andhra Pradesh , Telangana , Bihar , West Bengal , Gujarat and
Maharashtra are key contributors to Indian agriculture .
</item 4>

<item 5>
Agriculture in India
Amul : an integrated dairy with milk processing plant in Gujarat state . India has some of the world 's best agricultural yields in its tea plantations . A tea
estate in Kerala state . As of 2011 , India had a large and diverse agricultural sector , accounting , on average , for about 16 % of GDP and 10 % of export
earnings . India 's arable land area of 159.7 million hectares ( 394.6 million acres ) is the second largest in the world , after the United States . Its gross
irrigated crop area of 82.6 million hectares ( 215.6 million acres ) is the largest in the world . India is among the top three global producers of many crops ,
including wheat , rice , pulses , cotton , peanuts , fruits and vegetables . Worldwide , as of 2011 , India had the largest herds of buffalo and cattle , is the
largest producer of milk and has one of the largest and fastest growing poultry industries . Major crops and yields ( edit ) The following table presents the 20
most important agricultural products in India , by economic value , in 2009 .
</item 5>

<item 6>
Agriculture in India
The Indian farmer receives just 10 % to 23 % of the price the Indian consumer pays for exactly the same produce , the difference going to losses ,
inefficiencies and middlemen . Farmers in developed economies of Europe and the United States receive 64 % to 81 % . Productivity ( edit ) Although India has
attained self - sufficiency in food staples , the productivity of its farms is below that of Brazil , the United States , France and other nations . Indian
wheat farms , for example , produce about a third of the wheat per hectare per year compared to farms in France . Rice productivity in India was less than half
that of China . Other staples productivity in India is similarly low . Indian total factor productivity growth remains below 2 % per annum ; in contrast , China
's total factor productivity growths is about 6 % per annum , even though China also has smallholding farmers . Several studies suggest India could eradicate
its hunger and malnutrition and be a major source of food for the world by achieving productivity comparable with other countries . By contrast , Indian farms
in some regions post the best yields , for sugarcane , cassava and tea crops .
</item 6>

<item 7>
Agriculture in India
Bihar Peasant Life . Bengal Secretariat Press , Calcutta . References ( edit ) Jump up ^ https://books.google.com/books?id=pU6dCgAAQBAJ&pg=PA96 Jump up ^
Agriculture 's share in GDP declines to 13.7 % in 2012 - 13 Jump up ^ `` CIA Factbook : India '' . CIA Factbook . Central Intelligence Agency . Archived from
the original on 11 June 2008 . Retrieved 2008 - 06 - 10 . Jump up ^ Staff , India Brand Equity Foundation Agriculture and Food in India Accessed 7 May 2013 ^
Jump up to : India 's Agricultural Exports Climb to Record High , United States Department of Agriculture ( 2014 ) Jump up ^
http://www.ibef.org/exports/agriculture-and-food-industry-india.aspx Jump up ^ http://www.agriexam.com/?m=1 ^ Jump up to : `` FAOSTAT , 2010 data '' .
Faostat.fao.org . Retrieved 2011 - 09 - 17 .
</item 7>

<item 8>
Agriculture in India
Contents ( hide ) 1 Overview 2 History 2.1 Agriculture and colonialism 2.2 Indian agriculture after independence 3 Irrigation 4 Output 4.1 Major crops and
yields 4.2 Horticulture 4.3 Organic Agriculture 5 Problems 5.1 Infrastructure 5.2 Productivity 5.3 Farmer suicides 5.4 Diversion of agricultural land for non
agricultural purpose 6 Initiatives 7 Maps 8 See also 9 Bibliography 10 References 11 Further reading 12 External links Overview ( edit ) Rice Paddy Fields in
Tamil Nadu As per the 2010 FAO world agriculture statistics India is the world 's largest producer of many fresh fruits and vegetables , milk , major spices ,
select fibrous crops such as jute , staples such as millets and castor oil seed . India is the second largest producer of wheat and rice , the world 's major
food staples . India is the world 's second or third largest producer of several dry fruits , agriculture - based textile raw materials , roots and tuber crops
, pulses , farmed fish , eggs , coconut , sugarcane and numerous vegetables .
</item 8>

<item 9>
Agriculture in India
India 's integration into the global economy has been accompanied by impressive economic growth that has brought significant economic and social benefits to the
country . Nevertheless , disparities in income and human development are on the rise . Preliminary estimates suggest that in 2009 - 10 the combined all India
poverty rate was 32 % compared to 37 % in 2004 - 05 . Going forward , it will be essential for India to build a productive , competitive , and diversified
agricultural sector and facilitate rural , non-farm entrepreneurship and employment . Encouraging policies that promote competition in agricultural marketing
will ensure that farmers receive better prices . '' -- World Bank : `` India Country Overview 2011 '' A 2003 analysis of India 's agricultural growth from 1970
to 2001 by the Food and Agriculture Organisation identified systemic problems in Indian agriculture .
</item 9>

<item 10>
Agriculture in India
India 's agricultural economy is undergoing structural changes . Between 1970 and 2011 , the GDP share of agriculture has fallen from 43 % to 16 % . This is n't
because of reduced importance of agriculture or a consequence of agricultural policy . This is largely because of the rapid economic growth in services ,
industrial output , and non-agricultural sectors in India between 2000 and 2010 . Agricultural scientist MS Swaminathan has played a vital role in the green
revolution . In 2013 NDTV awarded him as 25 living legend of India for outstanding contribution to agriculture and making India a food sovereign country . An
irrigation canal in Gujarat . Irrigation contributes significantly to agriculture in India . Two states , Sikkim and Kerala have planned to shift fully to
organic farming by 2015 and 2016 respectively . Irrigation ( edit ) Main article : Irrigation in India Indian irrigation infrastructure includes a network of
major and minor canals from rivers , groundwater well - based systems , tanks , and other rainwater harvesting projects for agricultural activities . Of these ,
the groundwater system is the largest .
</item 10>
=== END ITEM id=1e2b9a5865a9b1cb ===

=== ITEM id=0ac235de6c976880 ===
Gold sentences and facts:
Sentence 0: Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1
  Fact 0.0: Spain is listed with the value 430.
  Fact 0.1: France is listed with the value 128.
  Fact 0.2: Italy is listed with the value 104.
  Fact 0.3: Russia is listed with the value 100.
  Fact 0.4: Portugal is listed with the value 87.
  Fact 0.5: Germany is listed with the value 77.
  Fact 0.6: United Kingdom is listed with the value 64.
  Fact 0.7: Poland is listed with the value 47.
  Fact 0.8: Greece is listed with the value 46.
  Fact 0.9: Turkey is listed with the value 41.
  Fact 0.10: Belgium is listed with the value 31.
  Fact 0.11: Netherlands is listed with the value 28.
  Fact 0.12: Romania is listed with the value 24.
  Fact 0.13: Switzerland is listed with the value 20.
  Fact 0.14: Austria is listed with the value 14.
  Fact 0.15: Sweden is listed with the value 12.
  Fact 0.16: Croatia is listed with the value 10.
  Fact 0.17: Ireland is listed with the value 9.
  Fact 0.18: Ukraine is listed with the value 9.
  Fact 0.19: Hungary is listed with the value 8.
  Fact 0.20: Bulgaria is listed with the value 6.
  Fact 0.21: Cyprus is listed with the value 6.
  Fact 0.22: Czech Republic is listed with the value 6.
  Fact 0.23: Finland is listed with the value 5.
  Fact 0.24: Lithuania is listed with the value 5.
  Fact 0.25: Norway is listed with the value 5.
  Fact 0.26: Serbia is listed with the value 5.
  Fact 0.27: Slovenia is listed with the value 5.
  Fact 0.28: Denmark is listed with the value 4.
  Fact 0.29: Latvia is listed with the value 4.
  Fact 0.30: Estonia is listed with the value 3.
  Fact 0.31: Azerbaijan is listed with the value 3.
  Fact 0.32: Bosnia and Herzegovina is listed with the value 3.
  Fact 0.33: Georgia is listed with the value 3.
  Fact 0.34: Luxembourg is listed with the value 3.
  Fact 0.35: Slovakia is listed with the value 3.
  Fact 0.36: Armenia is listed with the value 2.
  Fact 0.37: Albania is listed with the value 1.
  Fact 0.38: Andorra is listed with the value 1.
  Fact 0.39: Belarus is listed with the value 1.
  Fact 0.40: Iceland is listed with the value 1.
  Fact 0.41: Macedonia is listed with the value 1.
  Fact 0.42: Malta is listed with the value 1.
  Fact 0.43: Monaco is listed with the value 1.
  Fact 0.44: Montenegro is listed with the value 1.
Sentence 1: United Kingdom : 64
  Fact 1.0: United Kingdom is listed with the value 64.
Sentence 2: Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32 Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20 Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait : 5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1
  Fact 2.0: South Africa is listed under Africa with the value 9.
  Fact 2.1: Egypt is listed under Africa with the value 5.
  Fact 2.2: Morocco is listed under Africa with the value 4.
  Fact 2.3: Tunisia is listed under Africa with the value 3.
  Fact 2.4: Algeria is listed under Africa with the value 1.
  Fact 2.5: United States is listed under Americas with the value 85.
  Fact 2.6: Mexico is listed under Americas with the value 84.
  Fact 2.7: Brazil is listed under Americas with the value 56.
  Fact 2.8: Canada is listed under Americas with the value 32.
  Fact 2.9: Colombia is listed under Americas with the value 14.
  Fact 2.10: Argentina is listed under Americas with the value 10.
  Fact 2.11: Chile is listed under Americas with the value 9.
  Fact 2.12: Venezuela is listed under Americas with the value 8.
  Fact 2.13: Peru is listed under Americas with the value 4.
  Fact 2.14: Puerto Rico is listed under Americas with the value 4.
  Fact 2.15: Dominican Republic is listed under Americas with the value 3.
  Fact 2.16: Guatemala is listed under Americas with the value 3.
  Fact 2.17: Panama is listed under Americas with the value 3.
  Fact 2.18: Costa Rica is listed under Americas with the value 2.
  Fact 2.19: Ecuador is listed under Americas with the value 2.
  Fact 2.20: El Salvador is listed under Americas with the value 2.
  Fact 2.21: Honduras is listed under Americas with the value 2.
  Fact 2.22: Uruguay is listed under Americas with the value 2.
  Fact 2.23: Aruba is listed under Americas with the value 1.
  Fact 2.24: Nicaragua is listed under Americas with the value 1.
  Fact 2.25: Paraguay is listed under Americas with the value 1.
  Fact 2.26: China is listed under Asia with the value 200.
  Fact 2.27: Japan is listed under Asia with the value 99.
  Fact 2.28: South Korea is listed under Asia with the value 43.
  Fact 2.29: Saudi Arabia is listed under Asia with the value 39.
  Fact 2.30: Israel is listed under Asia with the value 24.
  Fact 2.31: India is listed under Asia with the value 20.
  Fact 2.32: Indonesia is listed under Asia with the value 17.
  Fact 2.33: Hong Kong is listed under Asia with the value 14.
  Fact 2.34: Thailand is listed under Asia with the value 11.
  Fact 2.35: United Arab Emirates is listed under Asia with the value 11.
  Fact 2.36: Malaysia is listed under Asia with the value 10.
  Fact 2.37: Singapore is listed under Asia with the value 9.
  Fact 2.38: Taiwan is listed under Asia with the value 9.
  Fact 2.39: Philippines is listed under Asia with the value 8.
  Fact 2.40: Lebanon is listed under Asia with the value 7.
  Fact 2.41: Kazakhstan is listed under Asia with the value 5.
  Fact 2.42: Kuwait is listed under Asia with the value 5.
  Fact 2.43: Qatar is listed under Asia with the value 5.
  Fact 2.44: Jordan is listed under Asia with the value 3.
  Fact 2.45: Bahrain is listed under Asia with the value 2.
  Fact 2.46: Macau is listed under Asia with the value 2.
  Fact 2.47: Oman is listed under Asia with the value 1.
  Fact 2.48: Vietnam is listed under Asia with the value 1.
  Fact 2.49: Spain is listed under Europe with the value 430.
  Fact 2.50: France is listed under Europe with the value 128.
  Fact 2.51: Italy is listed under Europe with the value 104.
  Fact 2.52: Russia is listed under Europe with the value 100.
  Fact 2.53: Portugal is listed under Europe with the value 87.
  Fact 2.54: Germany is listed under Europe with the value 77.
  Fact 2.55: United Kingdom is listed under Europe with the value 64.
  Fact 2.56: Poland is listed under Europe with the value 47.
  Fact 2.57: Greece is listed under Europe with the value 46.
  Fact 2.58: Turkey is listed under Europe with the value 41.
  Fact 2.59: Belgium is listed under Europe with the value 31.
  Fact 2.60: Netherlands is listed under Europe with the value 28.
  Fact 2.61: Romania is listed under Europe with the value 24.
  Fact 2.62: Switzerland is listed under Europe with the value 20.
  Fact 2.63: Austria is listed under Europe with the value 14.
  Fact 2.64: Sweden is listed under Europe with the value 12.
  Fact 2.65: Croatia is listed under Europe with the value 10.
  Fact 2.66: Ireland is listed under Europe with the value 9.
  Fact 2.67: Ukraine is listed under Europe with the value 9.
  Fact 2.68: Hungary is listed under Europe with the value 8.
  Fact 2.69: Bulgaria is listed under Europe with the value 6.
  Fact 2.70: Cyprus is listed under Europe with the value 6.
  Fact 2.71: Czech Republic is listed under Europe with the value 6.
  Fact 2.72: Finland is listed under Europe with the value 5.
  Fact 2.73: Lithuania is listed under Europe with the value 5.
  Fact 2.74: Norway is listed under Europe with the value 5.
  Fact 2.75: Serbia is listed under Europe with the value 5.
  Fact 2.76: Slovenia is listed under Europe with the value 5.
  Fact 2.77: Denmark is listed under Europe with the value 4.
  Fact 2.78: Latvia is listed under Europe with the value 4.
  Fact 2.79: Estonia is listed under Europe with the value 3.
  Fact 2.80: Azerbaijan is listed under Europe with the value 3.
  Fact 2.81: Bosnia and Herzegovina is listed under Europe with the value 3.
  Fact 2.82: Georgia is listed under Europe with the value 3.
  Fact 2.83: Luxembourg is listed under Europe with the value 3.
  Fact 2.84: Slovakia is listed under Europe with the value 3.
  Fact 2.85: Armenia is listed under Europe with the value 2.
  Fact 2.86: Albania is listed under Europe with the value 1.
  Fact 2.87: Andorra is listed under Europe with the value 1.
  Fact 2.88: Belarus is listed under Europe with the value 1.
  Fact 2.89: Iceland is listed under Europe with the value 1.
  Fact 2.90: Macedonia is listed under Europe with the value 1.
  Fact 2.91: Malta is listed under Europe with the value 1.
  Fact 2.92: Monaco is listed under Europe with the value 1.
  Fact 2.93: Montenegro is listed under Europe with the value 1.
  Fact 2.94: Australia is listed under Oceania with the value 19.
  Fact 2.95: New Zealand is listed under Oceania with the value 1.

Retrieved items:
<item 1>
Zara (retailer)
Zara ( retailer ) - Wikipedia Zara ( retailer ) Jump to : navigation , search Zara España , S.A. Type Sociedad Anónima Industry Retail Founded Arteixo , Galicia
( 24 May 1974 ; 43 years ago ( 1974 - 05 - 24 ) ) Founder Amancio Ortega Headquarters Arteixo , Galicia Number of locations 2266 stores Area served Worldwide
Products Clothing Revenue US $ 15.9 billion ( 2016 ) Parent Inditex Website zara.com Zara ( Spanish : ( ˈθaɾa ) ) is a Galician fast fashion ( clothing and
accessories ) retailer based in Arteixo , Galicia . The company was founded in 1975 by Amancio Ortega and Rosalía Mera . It is the main brand of the Inditex
group , the world 's largest apparel retailer . The fashion group also owns brands such as Massimo Dutti , Pull&Bear , Bershka , Stradivarius , Oysho , Zara
Home , and Uterqüe .
</item 1>

<item 2>
Zara (retailer)
Zara as of 2017 manages up to 20 clothing collections a year . Contents ( hide ) 1 History 1.1 Early history 1.2 Expansion 2 Products 3 Manufacturing and
distribution 3.1 Non-toxic clothing 4 Controversy 5 Stores 6 See also 7 References 8 External links History ( edit ) Early history ( edit ) Amancio Ortega
opened the first Zara store in 1975 in downtown A Coruña , Galicia , Spain . Ortega initially named the store Zorba after the classic film Zorba the Greek , but
after learning there was a bar with the same name two blocks away , they rearranged the letters molded for the sign to ' Zara . ' It is believed the extra ' a '
came from an additional set of letters that had been made for the company . The first store featured low - priced lookalike products of popular , higher - end
clothing fashions . Ortega opened additional stores throughout Spain .
</item 2>

<item 3>
Zara (retailer)
Products ( edit ) Zara stores have men 's and women 's clothing , as well as children 's clothing ( Zara Kids ) . Zara 's products are supplied based on
consumer trends . Its highly responsive supply chain ships new products to stores twice a week . After products are designed , they take ten to fifteen days to
reach the stores . All of the clothing is processed through the distribution center in Spain . New items are inspected , sorted , tagged , and loaded into
trucks . In most cases , the clothing is delivered within 48 hours . Zara produces over 450 million items per year . Manufacturing and distribution ( edit )
Zara Almere - Heren The men 's department of a typical Zara store . Almere , Netherlands Zara clothing made in Portugal Zara store in New York City . Zara store
in Brisbane Zara store in Palma de Mallorca ( Spain ) . Zara store in Brussels ( Belgium ) . Reportedly , Zara needs just one week to develop a new product and
get it to stores , compared to the six - month industry average , and launches around 12,000 new designs each year .
</item 3>

<item 4>
Zara (retailer)
: 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24
Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5
Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia
: 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1 See also ( edit )
Spain portal Companies portal Fashion portal Bonds ( clothing ) Gap Inc . H&M List of companies of Spain European Retail Round Table References ( edit ) Jump up
^ `` Zara '' . inditex.com .
</item 4>

<item 5>
Zara (retailer)
When the news was eventually picked up by media outlets , and Inditex , Zara 's owning company , was asked to comment on the issue , Inditex replied that the
items in question have been suspended from sale , and that they are in contact with Bassen 's lawyer to clarify and address the issue . In November 2017 ,
customers shopping at Zara stores in Istanbul , Turkey found handwritten notes purportedly from Turkish workers in the pockets of in - store garments asking
shoppers to pressure Zara into paying them the wages they claim they are owed . Stores ( edit ) There are over 2,100 Zara stores located across 88 countries .
Zara usually selects the best located and most expensive real estate locations in the world to open its flagship stores . Zara 's flagship stores : Fifth Avenue
, New York Oxford Street , London Calle Serrano , Madrid Via del Corso , Rome Champs - Élysées , Paris Oscar Freire Street , Sao Paulo Corso Vittorio Emanuele ,
Milan Nevsky Prospect , Saint Petersburg Shibuya and Ginza , Tokyo Myeongdong in Seoul Flora Fountain in Mumbai The number of Zara stores in each country ,
</item 5>

<item 6>
Zara (retailer)
Zara has a policy of zero advertising ; the company preferred to invest a percentage of revenues in opening new stores instead . Zara set up its own factory in
La Coruña ( a city known for its textile industry ) in 1980 , and upgraded to reverse milk - run - type production and distribution facilities in 1990 . This
approach , designed by Toyota Motor Corp. , was called the just - in - time ( JIT ) system . It enabled the company to establish a business model that allows
self - containment throughout the stages of materials , manufacture , product completion and distribution to stores worldwide within just a few days . Most of
the products Zara sells are manufactured in proximity countries like Spain , Portugal , Turkey and Morocco . While some competitors outsource all production to
Asia , Zara manufactures its most fashionable items -- half of all its merchandise -- at a dozen company - owned factories in Spain and Portugal and Turkey ,
particularly in Galicia and northern Portugal and Turkey . Clothes with a longer shelf life , such as basic T - shirts , are outsourced to low - cost suppliers
, mainly in Asia .
</item 6>

<item 7>
Zara (retailer)
Ortega opened additional stores throughout Spain . During the 1980s , Ortega changed the design , manufacturing , and distribution process to reduce lead times
and react to new trends in a quicker way , which he called `` instant fashions '' . The improvements included the use of information technologies and using
groups of designers instead of individuals . Zara 's previous logo Expansion ( edit ) In 1988 , the company started its international expansion through Porto ,
Portugal . In 1989 , it entered the United States , and then France in 1990 . During the 1990s , Zara expanded to Mexico ( 1992 ) , Greece , Belgium and Sweden
( 1993 ) . In the early 2000s , Zara opened its first stores in Japan and Singapore ( 2002 ) , Russia and Malaysia ( 2003 ) , China , Morocco , Estonia ,
Hungary and Romania ( 2004 ) , the Philippines , Costa Rica and Indonesia ( 2005 ) , South Korea ( 2008 ) , India ( 2010 ) , and South Africa and Australia (
2011 ) .
</item 7>

<item 8>
Zara (retailer)
as of 31 November 2017 : Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32
Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El
Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20
Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait :
5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United
</item 8>

<item 9>
Zara is a Galician fast-fashion retailer founded by Amancio Ortega in A Coruña, Spain, in 1975, and Inditex’s flagship brand. It sells women’s, men’s and
children’s clothing worldwide. Its rapid, trend-responsive supply chain introduces new designs frequently, with much production near Spain and distribution
through a central hub. Zara expanded internationally and online, using RFID inventory technology. The article reports more than 2,100 stores across 88 countries
</item 9>

<item 10>
Zara (retailer)
On September 2010 , Zara launched its online boutique . The website began in Spain , the UK , Portugal , Italy , Germany and France . In November that same year
, Zara Online extended the service to five more countries : Austria , Ireland , the Netherlands , Belgium and Luxembourg . Online stores began operating in the
United States in 2011 , Russia and Canada in 2013 , and Mexico , Romania , and South Korea in 2014 . . India In 4 october 2017 . Zara introduced the use of RFID
technology in its stores in 2014 . The RFID chips are located in the security tags which are removed from clothing when it is purchased and can be reused . The
chip allows the company to quickly take inventory by detecting radio signals from the RFID tags . When an item is sold , the stockroom is immediately notified
so that the item can be replaced . An item that is not on the shelf can easily be found with the RFID tag . In 2015 , Zara was ranked 30 on Interbrand 's list
of best global brands .
</item 10>
=== END ITEM id=0ac235de6c976880 ===

=== ITEM id=9fc7f819c10c03a1 ===
Gold sentences and facts:
Sentence 0: The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet Bandits and must outrun and out - prank them again .
  Fact 0.0: Kevin and his family decide to take a trip to Florida.
  Fact 0.1: Kevin takes the wrong plane.
  Fact 0.2: Kevin ends up in New York City.
  Fact 0.3: Kevin tries to make do with what he has.
  Fact 0.4: Kevin uses his father's credit card to stay at the Plaza Hotel.
  Fact 0.5: Kevin is soon confronted by the Wet Bandits.
  Fact 0.6: Kevin must outrun and out-prank the Wet Bandits again.

Retrieved items:
<item 1>
Home Alone 2: Lost in New York
Home Alone 2 : Lost in New York - wikipedia Home Alone 2 : Lost in New York Jump to : navigation , search `` Home Alone 2 '' redirects here . For the video game
based on this film , see Home Alone 2 ( video game ) . Home Alone 2 : Lost in New York Theatrical release poster Directed by Chris Columbus Produced by John
Hughes Written by John Hughes Based on Characters by John Hughes Starring Macaulay Culkin Joe Pesci Daniel Stern John Heard Tim Curry Brenda Fricker Catherine
O'Hara Music by John Williams Cinematography Julio Macat Edited by Raja Gosnell Production company Hughes Entertainment Distributed by 20th Century Fox Release
date November 20 , 1992 ( 1992 - 11 - 20 ) Running time 120 minutes Country United States Language English Budget $20 million Box office $359 million Home Alone
2 : Lost in New York is a 1992 American Christmas comedy film written and produced by John Hughes and directed by Chris Columbus . It is the second film in the
Home Alone series and the sequel to Home Alone .
</item 1>

<item 2>
Home Alone 2: Lost in New York
It is the second film in the Home Alone series and the sequel to Home Alone . Macaulay Culkin reprises his role as Kevin McCallister , while Joe Pesci and
Daniel Stern reprise their roles as the Wet Bandits , now known as the Sticky Bandits . Catherine O'Hara , John Heard , Rob Schneider , Tim Curry , and Brenda
Fricker are also featured . The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New
York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet
Bandits and must outrun and out - prank them again . The film was shot in Winnetka , Illinois ; O'Hare International Airport in Chicago ; Evanston , Illinois ;
and New York City ( which was star Culkin 's hometown at the time ) . The exterior of Duncan 's Toy Chest in New York City was filmed outside of the Rookery
Building in downtown Chicago .
</item 2>

<item 3>
Home Alone 2: Lost in New York
Meanwhile , Kevin goes to his uncle Rob 's townhouse , only to find the house vacant and undergoing renovations , while Rob and his family are in Paris . At
Central Park , he encounters and befriends the pigeon lady . They go to Carnegie Hall , where an orchestra is performing . The pigeon lady explains how her life
collapsed and how she dealt with it by taking care of the pigeons in the park . Kevin gives the pigeon lady some advice and promises that he will be her friend
. Kevin , after remembering what the bandits said , returns to the townhouse , and rigs it with numerous booby traps . Kevin arrives at the toy store during
Harry and Marv 's robbery and throws a brick through the window , setting off the store 's alarm . Kevin then lures the duo to the townhouse , where he springs
the traps and Harry and Marv suffer various injuries . When the duo chase Kevin around the townhouse , he escapes and calls the police .
</item 3>

<item 4>
Home Alone 2: Lost in New York
Duncan . Kevin learns that the proceeds from the store 's Christmas sales will be donated to a children 's hospital . Kevin donates $20 to the hospital . As a
token of appreciation , Mr. Duncan offers Kevin a pair of ceramic turtledoves as a gift , instructing him to give one to another person as a sign of eternal
friendship . After running into Harry and Marv , who have recently escaped from prison and are now called `` The Sticky Bandits '' , Kevin retreats to the Plaza
Hotel . The hotel 's concierge , Mr. Hector , confronts Kevin about the credit card , which has been reported stolen . Kevin flees after evading Mr. Hector ,
but is captured by Harry and Marv . The duo discuss plans for breaking into the toy store that night . Kevin manages to escape by pinching a woman in front of
him , which results in the woman attacking the duo . Kevin 's family travels to New York after tracking the whereabouts of the stolen credit card , and Kate
searches the city for Kevin .
</item 4>

<item 5>
Home Alone 2: Lost in New York
Peter suddenly calls out , `` Kevin , you spent $967 on room service ? ! '' at which point Kevin runs off . Cast ( edit ) Main cast ( edit ) Macaulay Culkin as
Kevin McCallister Joe Pesci as Harry Lime Daniel Stern as Marv Merchants The McCallister family ( edit ) Catherine O'Hara as Kate McCallister John Heard as
Peter McCallister Devin Ratray as Buzz McCallister , Kevin 's brother who gets him into trouble Kieran Culkin ( Macaulay Culkin 's younger brother ) as Fuller
McCallister Gerry Bamman as Frank McCallister , Kevin 's uncle Hillary Wolf as Megan McCallister Maureen Elisabeth Shay ( who replaces Angela Goethals ) as
Linnie McCallister Mike Maronna as Jeff McCallister Terrie Snell as Leslie McCallister , Kevin 's aunt Jedidiah Cohen as Rod McCallister Senta Moses as Tracy
McCallister Daiana Campeanu as Sondra McCallister Anna Slotky as Brooke McCallister Other characters ( edit ) Brenda Fricker as the Pigeon Lady Tim Curry as Mr.
</item 5>

<item 6>
Home Alone 2: Lost in New York
In the beginning of the novelization , a prologue , which ends up being Marv 's nightmare in prison , he and Harry sneak away from the cops and return to Kevin
's house to seek revenge on Kevin . Kevin bolts into the garage with Marv and Harry in hot pursuit . Harry and Marv end up triggering extra traps that Kevin had
set up in the garage . Kevin watches as Marv ends up triggering a trap where a running lawnmower falls on his head ( This was a trap featured in Home Alone 3 )
. Sequels ( edit ) Main article : Home Alone 3 A second sequel , Home Alone 3 , followed in 1997 . Two additional television movie sequels , Taking Back the
House and The Holiday Heist , would follow in 2002 and 2012 , respectively . See also ( edit ) List of films featuring fictional films References ( edit ) Jump
up ^ `` HOME ALONE 2 - LOST IN NEW YORK ( PG ) '' . British Board of Film Classification . November 4 , 2017 . Retrieved November 10 , 2015 .
</item 6>

<item 7>
Home Alone 2: Lost in New York
Contents ( hide ) 1 Plot 2 Cast 2.1 Main cast 2.2 The McCallister family 2.3 Other characters 3 Release 3.1 Marketing 3.2 Box office 3.3 Critical reception 3.4
Home media 3.5 Social Media 4 Music 4.1 Original Score 4.2 Original Soundtrack Album 4.3 The Deluxe Edition 5 Novelization 6 Sequels 7 See also 8 References 9
External links Plot ( edit ) In Chicago , the McCallister family is preparing for a Christmas vacation in Miami . On the night before their departure , the
entire family gathers at Peter and Kate 's home , where their youngest son , Kevin , sees Florida as contradictory to Christmas , on his opinion of the lack of
Christmas trees in Florida . During the school Christmas pageant , Kevin 's older brother Buzz humiliates him during his solo , causing Kevin to retaliate ,
thereby ruining the performance .
</item 7>

<item 8>
Home Alone 2: Lost in New York
While Buzz fakes an apology to Kevin in front of the McCallisters ( who are convinced it is sincere ) , Kevin refuses to apologize to Buzz and storms to the
third floor of the house . Prior to the pageant , Peter has unknowingly caused the alarm clock to reset ; consequently , the family once again oversleeps . In
the confusion they have to rush to reach the airport on time , Kevin falls behind everyone else , and mistakenly boards a flight bound for New York City while
trying to replace the batteries for a Talkboy , carrying Peter 's bag containing his wallet and a large amount of money ; upon arrival in Miami , Kate realizes
that Kevin is missing again . In New York , Kevin tours the city and meets a homeless woman tending to pigeons at Central Park , where he runs off . Kevin goes
to the Plaza Hotel , where he tantalizes the staff into checking in with Peter 's credit card . On Christmas Eve , Kevin tours the city in a limousine and
visits a toy store where he meets its philanthropic owner , Mr. Duncan .
</item 8>

<item 9>
Home Alone 2: Lost in New York
When the duo chase Kevin around the townhouse , he escapes and calls the police . Harry and Marv catch him and take him down to Central Park to kill him , but
the pigeon lady sneaks in and throws birdseed on the duo , and the pigeons storm in to incapacitate them . Kevin sets off fireworks he had bought earlier to
signal the police . The police arrive and arrest the bandits . At the toy store , Mr. Duncan finds a note from Kevin attached to the brick , explaining about
what happened . Kate , still searching for Kevin on foot , remembers his fondness for Christmas trees . After observing Kevin making a wish at the Rockefeller
Center Christmas Tree , Kate meets him there , and they reconcile . On Christmas Day , a truckload of gifts arrive at the McCallisters ' hotel room from the toy
store . Kevin and Buzz reconcile , and Buzz allows him to open up the first present . Kevin goes to Central Park to give the pigeon lady the second turtledove .
At the Plaza , Buzz receives the bill for Kevin 's stay from Cedric and shows it to Peter .
</item 9>

<item 10>
Home Alone 2: Lost in New York
External links ( edit ) Wikiquote has quotations related to : Home Alone 2 : Lost in New York Home Alone 2 : Lost in New York on IMDb Home Alone 2 : Lost in New
York at the TCM Movie Database Home Alone 2 : Lost in New York at AllMovie Home Alone 2 : Lost in New York at Rotten Tomatoes Movie Locations Guide.com - Maps
and directions to Home Alone 2 Filming Locations Home Alone Films Home Alone ( 1990 ) Home Alone 2 : Lost in New York ( 1992 ) Home Alone 3 ( 1997 ) Home Alone
4 ( 2002 ) Home Alone : The Holiday Heist ( 2012 ) Video games Home Alone ( 1991 ) Home Alone 2 : Lost in New York ( 1992 ) Home Alone ( 2006 ) Miscellaneous
Home Alone : Original Motion Picture Soundtrack Talkboy Chris Columbus Directed Adventures in Babysitting ( 1987 ) Heartbreak Hotel ( 1988 ) Home Alone ( 1990 )
Only the Lonely ( 1991 ) Home Alone 2 : Lost in New York ( 1992 ) Mrs.
</item 10>
=== END ITEM id=9fc7f819c10c03a1 ===
