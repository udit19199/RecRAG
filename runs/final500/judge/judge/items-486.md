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

This file has 11 items.

=== ITEM id=afaa6e8b678681e8 ===
Gold sentences and facts:
Sentence 0: ", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)
  Fact 0.0: This person did voice work for My Life As A Teenage Robot.
  Fact 0.1: This person did voice work for The Replacements.
  Fact 0.2: This person did the voice of Knotty in a 1992 work whose title is blank in the sentence.
  Fact 0.3: This person did additional voices in the film Fantasia 2000 (1999).
  Fact 0.4: This person did additional voices in the film Sing (2016).
Sentence 1:  It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.
  Fact 1.0: The film was directed by Garth Jennings.
  Fact 1.1: The film was written by Garth Jennings.
  Fact 1.2: The film was co-directed by Christophe Lourdelet.
  Fact 1.3: The film starred the voices of Matthew McConaughey.
  Fact 1.4: The film starred the voices of Reese Witherspoon.
  Fact 1.5: The film starred the voices of Seth MacFarlane.
  Fact 1.6: The film starred the voices of Scarlett Johansson.
  Fact 1.7: The film starred the voices of John C. Reilly.
  Fact 1.8: The film starred the voices of Taron Egerton.
  Fact 1.9: The film starred the voices of Tori Kelly.

Retrieved items:
<item 1>
Winnie the Pooh (2011 film)
Winnie the Pooh is a 2011 American animated buddy musical comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 51st Disney animated feature film.

 Inspired by A. A. Milne's stories of the same name, the film is part of Disney's "Winnie the Pooh" franchise, the fifth theatrical "Winnie the Pooh" film
released, and Walt Disney Animation Studios' second adaptation of "Winnie-the-Pooh" stories.

 Jim Cummings reprises his vocal roles as Winnie the Pooh and Tigger, while series newcomers Travis Oates, Tom Kenny, Craig Ferguson, Bud Luckey, and Kristen
Anderson-Lopez provide the voices of Piglet, Rabbit, Owl, Eeyore, and Kanga, respectively.

 In the film, the aforementioned residents of the Hundred Acre Wood embark on a quest to save Christopher Robin from an imaginary culprit while Pooh deals with
a hunger for honey.
</item 1>

<item 2>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 2>

<item 3>
Small Apartments
Small Apartments is a 2012 American comedy film directed by Jonas Åkerlund.

 It tells the story of Franklin Franklin, played by Matt Lucas, who by mistake kills his landlord, Mr. Olivetti, played by Peter Stormare.

 The cast co-stars Dolph Lundgren, Johnny Knoxville, James Caan, Billy Crystal, Juno Temple, Rebel Wilson, Saffron Burrows and Amanda Plummer.

 The screenplay was written by Chris Millis and adapted from his own novella.

 The film premiered at the South by Southwest Film Festival on March 10, 2012.
</item 3>

<item 4>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 4>

<item 5>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 5>

<item 6>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 6>

<item 7>
Are We Done Yet?
Are We Done Yet?

 is a 2007 American family comedy film starring Ice Cube.

 The film is a remake of the Cary Grant comedy "Mr. Blandings Builds His Dream House" and a sequel to the 2005 comedy "Are We There Yet?

" The film was directed by Steve Carr from a screenplay by Hank Nelken.

 It was produced by Revolution Studios and RKO Pictures and distributed by Columbia Pictures.

 The film was shot on location in Tsawwassen, British Columbia, Canada, but is set in Newberg, Oregon, United States.
</item 7>

<item 8>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 8>

<item 9>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 9>

<item 10>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 10>
=== END ITEM id=afaa6e8b678681e8 ===

=== ITEM id=e2a450d60e48f961 ===
Gold sentences and facts:
Sentence 0: Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in China and Japan.
  Fact 0.0: Pterostyrax is also called the epaulette tree.
  Fact 0.1: Pterostyrax is a small genus.
  Fact 0.2: Pterostyrax has four species.
  Fact 0.3: Pterostyrax species are deciduous.
  Fact 0.4: Pterostyrax species are large shrubs or small trees.
  Fact 0.5: Pterostyrax is in the family Styracaceae.
  Fact 0.6: Pterostyrax is native to eastern Asia.
  Fact 0.7: Pterostyrax is native to China.
  Fact 0.8: Pterostyrax is native to Japan.
Sentence 1: Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.
  Fact 1.0: Dregea is a genus of vines.
  Fact 1.1: Dregea is in the family Apocynaceae.
  Fact 1.2: Dregea was first described as a genus with this name in 1838.
Sentence 2:  It is native to Africa and southern Asia.
  Fact 2.0: Dregea is native to Africa.
  Fact 2.1: Dregea is native to southern Asia.

Retrieved items:
<item 1>
Pterostyrax hispidus
Pterostyrax hispidus, the epaulette tree or fragrant epaulette tree, is a species of flowering plant in the family Styracaceae, native to China and Japan.

 Growing to 15 m tall by 12 m broad, it is a substantial, spreading, deciduous shrub with oval leaves up to 20 cm long, and clusters of pure white, fragrant,
pendent flowers in summer.
</item 1>

<item 2>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 2>

<item 3>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 3>

<item 4>
Dregea sinensis
Dregea sinensis is a species of plant in the Apocynaceae family that is native to China.
</item 4>

<item 5>
Dregea
Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.

 It is native to Africa and southern Asia.
</item 5>

<item 6>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 6>

<item 7>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 7>

<item 8>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 8>

<item 9>
Styrax
Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm
temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.

 The resin obtained from the tree is called storax or benzoin (not to be confused with the "Liquidambar" storax balsam).
</item 9>

<item 10>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 10>
=== END ITEM id=e2a450d60e48f961 ===

=== ITEM id=29440372d1fe60d7 ===
Gold sentences and facts:
Sentence 0: The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.
  Fact 0.0: The Long–Allen Bridge is a truss bridge.
  Fact 0.1: The Long–Allen Bridge is in Shreveport, Louisiana.
  Fact 0.2: The Long–Allen Bridge is named for Louisiana governor Huey Long.
  Fact 0.3: The Long–Allen Bridge is named for Louisiana governor Oscar K. Allen.
Sentence 1:  Opened in 1933, it carries U.S. Route 80 across the Red River.
  Fact 1.0: The Long–Allen Bridge opened in 1933.
  Fact 1.1: The Long–Allen Bridge carries U.S. Route 80 across the Red River.
Sentence 2: U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.
  Fact 2.0: U.S. Route 80 (US 80) is an east–west United States Numbered Highway.
  Fact 2.1: Much of U.S. Route 80 was once part of the early auto trail known as the Dixie Overland Highway.
Sentence 3:  The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
  Fact 3.0: The eastern terminus of U.S. Route 80 is in Tybee Island, Georgia.
  Fact 3.1: The eastern terminus of U.S. Route 80 is at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street.
  Fact 3.2: The eastern terminus of U.S. Route 80 is near the Atlantic Ocean.

Retrieved items:
<item 1>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 1>

<item 2>
Pennsylvania Route 145
Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.

 It connects Interstate 78 (I-78) and PA 309 in Lanark, Lehigh County to PA 248 in Weiders Crossing, Northampton County.

 PA 145 is the main north-south arterial into Allentown, the third-largest city in the state.

 The route enters the city on South 4th Street and follows multiple streets to downtown, where it follows the one-way pair of 6th Street northbound and 7th
Street southbound.

 North of Allentown in Whitehall Township, a seven-mile (11.2 km) portion of PA 145 is known as MacArthur Road, named in honor of General Douglas MacArthur.

 MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the
remainder of the road a four-lane divided highway.

 MacArthur Road is the location of the main commercial center of the Lehigh Valley.
</item 2>

<item 3>
Rocky Creek Bridge No. 01089
Rocky Creek Bridge No. 01089, also known as Ben Jones Bridge, is a concrete highway bridge spanning Rocky Creek along the Pacific Ocean coast of the U.S. state
of Oregon.

 The bridge crosses a gorge near the creek's mouth, about 2 mi south of Depoe Bay in Lincoln County.

 Built in 1927, the bridge originally carried U.S. Route 101 (the Oregon Coast Highway) over the creek.

 After a stretch of the highway was relocated to make it straighter, the abandoned piece of the old highway became Otter Crest Loop Road, a bypass west of the
new highway.
</item 3>

<item 4>
U.S. Route 80
U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.

 As the "0" in the route number indicates, it was originally a cross-country route, from the Pacific Ocean to the Atlantic Ocean.

 However, the entire segment west of Dallas, Texas, has been decommissioned in favor of various Interstate Highways and state highways.

 Currently, the highway's western terminus is at an interchange with Interstate 30 (I-30) on the Dallas–Mesquite, Texas city line.

 The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
</item 4>

<item 5>
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 5>

<item 6>
Long–Allen Bridge (Shreveport)
The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.

 Opened in 1933, it carries U.S. Route 80 across the Red River.

 It is also known as the Texas Street Bridge.
</item 6>

<item 7>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 7>

<item 8>
Interstate 84 in Oregon
Interstate 84 (I-84) in the U.S. state of Oregon, also known as the Vietnam Veterans Memorial Highway, travels east–west, following the Columbia River and the
rough path of the old Oregon Trail from Portland east to Idaho.

 For this reason, it is also known as most of the Columbia River Highway No. 2 and all of the Old Oregon Trail Highway No. 6 (see Oregon highways and routes).

 It also follows along or near U.S. Route 30.

 The entire highway carried the designation Interstate 80N (I-80N, or Interstate 80 North) until 1980, when this was changed to I-84.
</item 8>

<item 9>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 9>

<item 10>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 10>
=== END ITEM id=29440372d1fe60d7 ===

=== ITEM id=efca005e355f188c ===
Gold sentences and facts:
Sentence 0: The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
  Fact 0.0: The Consul is an opera.
  Fact 0.1: The Consul is an opera in three acts.
  Fact 0.2: The Consul has music by Gian Carlo Menotti.
  Fact 0.3: The Consul has a libretto by Gian Carlo Menotti.
  Fact 0.4: The Consul is Gian Carlo Menotti's first full-length opera.
Sentence 1:  He completed the music for the opera while living in Zurich in 1916.
  Fact 1.0: Gian Carlo Menotti completed the music for The Consul.
  Fact 1.1: Gian Carlo Menotti completed the music for The Consul in 1916.
  Fact 1.2: Gian Carlo Menotti was living in Zurich in 1916.
  Fact 1.3: Gian Carlo Menotti completed the music for The Consul while living in Zurich.

Retrieved items:
<item 1>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 1>

<item 2>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 2>

<item 3>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 3>

<item 4>
The Consul
The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
</item 4>

<item 5>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 5>

<item 6>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 6>

<item 7>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 7>

<item 8>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 8>

<item 9>
Commedia dell’arte, Italy’s 16th–18th-century professional masked theatre, combined scripted scenarios, improvisation, pantomime and comic lazzi. Stock figures
included Harlequin, Columbina, Pierrot, Pedrolino, Tartaglia and French clown-servant Gilles. Niccolò Barbieri (1586–1641), actor-writer known as Beltrame,
created Scapino; his play *L’inavertito* inspired M
</item 9>

<item 10>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 10>
=== END ITEM id=efca005e355f188c ===

=== ITEM id=1857d152ae588314 ===
Gold sentences and facts:
Sentence 0:  The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
  Fact 0.0: The ground was home to the Boston Reds.
  Fact 0.1: The Boston Reds played in the Players' League in 1890.
  Fact 0.2: The Boston Reds played in the American Association in 1891.
Sentence 1:  It is home to the Port of Boston, a major shipping facility in the northeastern United States.
  Fact 1.0: The ground is home to the Port of Boston.
  Fact 1.1: The Port of Boston is a major shipping facility.
  Fact 1.2: The Port of Boston is in the northeastern United States.

Retrieved items:
<item 1>
New Bedford Bay Sox
The New Bedford Bay Sox are a baseball team that plays in the New England Collegiate Baseball League, a collegiate summer baseball league located in the
northeastern United States region of New England.

 The team is located in New Bedford, Massachusetts.

 New Bedford's Paul Walsh Field serves as the home field of the Bay Sox.

 Th NECBL's players use wooden bats, as opposed to aluminum, and are college baseball players recruited from the top intercollegiate baseball programs in the
United States.

 The franchise has appeared in league finals four times in their history, all as the Torrington Twisters (1997, 1998, 2003, 2006), and have twice played host to
the NECBL All-Star Game as the Twisters (1998, 2008).

 Pat and Beth O'Connor, owners of Little Fenway and Little Wrigley joined the ownership team of the New Bedford Bay Sox in October, 2011, and signed a 3-year
lease to play at Paul Walsh Field.

 In November 2011, former MLB player Rick Miller was announced as the 2012 Bay Sox manager.
</item 1>

<item 2>
Transportation in Boston
The Boston transportation system includes roadway, subway, regional rail, air, and sea options for passenger and freight transit in Boston, Massachusetts.

 The Massachusetts Port Authority (Massport) operates the Port of Boston, which includes a container shipping facility in South Boston, and Logan International
Airport, in East Boston.

 The Massachusetts Bay Transportation Authority (MBTA) operates bus, subway, short distance rail, and water ferry passenger services throughout the city and
region.

 Amtrak operates passenger rail service to and from major northeastern cities.

 A major bus terminal at South Station is served by varied intercity bus companies.

 The city is bisected by major highways I-90 and I-93, the intersection of which has undergone a major renovation, nicknamed the Big Dig.
</item 2>

<item 3>
Braves Field
Braves Field was a baseball park in the Northeastern United States, located in Boston, Massachusetts.

 Today the site is home to Nickerson Field on the campus of Boston University.

 The stadium was home of the Boston Braves of the National League from 1915–1952, prior to the Braves' move to Milwaukee in 1953.

 The stadium hosted the 1936 Major League Baseball All-Star Game and Braves home games during the 1948 World Series.

 The Boston Red Sox used Braves Field for their home games in the 1915 and 1916 World Series since the stadium had a larger seating capacity than Fenway Park.

 Braves Field was the site of Babe Ruth's final season, playing for the Braves in 1935.

 From 1929 to 1932, the Boston Red Sox played select regular season games periodically at Braves Field.

 On May 1, 1926, Braves Field hosted the longest baseball game in history – 26 Innings.

 The game ended in a 1–1 tie.
</item 3>

<item 4>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 4>

<item 5>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 5>

<item 6>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 6>

<item 7>
New Bedford Bay Sox
Miller was succeeded as manager prior to the 2015 season by Westport, MA native and former Boston Red Sox draft pick Kyle Fernandes.

 Fernandes has led the Bay Sox to back-to-back postseason appearances in his two seasons as manager, most recently losing in a three-game series to the eventual
NECBL Champions the Mystic Schooners.
</item 7>

<item 8>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 8>

<item 9>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 9>

<item 10>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 10>
=== END ITEM id=1857d152ae588314 ===

=== ITEM id=1cf8c24f8e9b0d54 ===
Gold sentences and facts:
Sentence 0: Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.
  Fact 0.0: Eric Arthur Blair was born on 25 June 1903.
  Fact 0.1: Eric Arthur Blair died on 21 January 1950.
  Fact 0.2: Eric Arthur Blair's pen name was George Orwell.
  Fact 0.3: Eric Arthur Blair is better known as George Orwell.
  Fact 0.4: Eric Arthur Blair was an English novelist.
  Fact 0.5: Eric Arthur Blair was an English essayist.
  Fact 0.6: Eric Arthur Blair was an English journalist.
  Fact 0.7: Eric Arthur Blair was an English critic.
Sentence 1: Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.
  Fact 1.0: Nelson Richard DeMille was born on August 23, 1943.
  Fact 1.1: Nelson Richard DeMille is an American author.
  Fact 1.2: Nelson Richard DeMille writes action adventure novels.
  Fact 1.3: Nelson Richard DeMille writes suspense novels.
Sentence 2:  DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
  Fact 2.0: Nelson Richard DeMille has written under the pen name Jack Cannon.
  Fact 2.1: Nelson Richard DeMille has written under the pen name Kurt Ladner.
  Fact 2.2: Nelson Richard DeMille has written under the pen name Ellen Kay.
  Fact 2.3: Nelson Richard DeMille has written under the pen name Brad Matthews.

Retrieved items:
<item 1>
Wild Fire (novel)
Wild Fire is a 2006 novel by American author, Nelson DeMille.

 It is the fourth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to Night Fall and takes place approximately one year later.

 Wild Fire is followed by DeMille’s 2010 novel, The Lion.
</item 1>

<item 2>
The Lion's Game
The Lion's Game is a 2000 novel by American author Nelson DeMille.

 It is the second of DeMille's novels to feature the detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New
York City.

 The 2000 novel The Lion’s Game is the sequel to Plum Island and is followed by the 2004 book Night Fall.

 The book also briefly mentions events from other DeMille novels like "The Charm School" and "The Gold Coast", that aren't strictly part of the Corey-universe.
</item 2>

<item 3>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 3>

<item 4>
Nelson Algren (1909–1981), an American writer, published *The Man with the Golden Arm* in 1949; it won the 1950 National Book Award and was adapted into a 1955
film. His 1956 novel *A Walk on the Wild Side* is known for his three rules of life: avoid cards with “Doc,” food at “Mom’s,” and partners with worse troubles
than yours.

Nelson DeMille (born 1943
</item 4>

<item 5>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 5>

<item 6>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 6>

<item 7>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 7>

<item 8>
The Panther (novel)
The Panther is a 2012 novel by American author Nelson DeMille.

 It is the sixth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "The Lion".

 The Panther is followed by DeMille’s 2015 novel, "Radiant Angel".

 Also featured in this novel is DeMille’s other fictional character, Paul Brenner, who appears in The General's Daughter and Up Country.
</item 8>

<item 9>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 9>

<item 10>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 10>
=== END ITEM id=1cf8c24f8e9b0d54 ===

=== ITEM id=dbc276807de5cc07 ===
Gold sentences and facts:
Sentence 0: "Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series overall.
  Fact 0.0: Moho House is the twenty-first episode of the twenty-eighth season of the animated television series The Simpsons.
  Fact 0.1: Moho House is the 617th episode of The Simpsons overall.
  Fact 0.2: The Simpsons is an animated television series.
Sentence 1:  On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
  Fact 1.0: Fox announced on May 4, 2015.
  Fact 1.1: Fox announced that The Simpsons had been renewed for season 28.

Retrieved items:
<item 1>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 1>

<item 2>
List of The Simpsons comics
Since then, many more titles have been published, out of which "Simpsons Comics", "Bart Simpson", "Bart Simpson's Treehouse of Horror", "Simpsons Super
Spectacular", Simpsons Summer Shindig, and "Simpsons Winter Wingding".
</item 2>

<item 3>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 3>

<item 4>
The Simpsons (season 28)
The twenty-eighth season of the animated television series "The Simpsons" began airing on Fox in the United States on September 25, 2016, and ended on May 21,
2017.

 On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
</item 4>

<item 5>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 5>

<item 6>
List of The Simpsons comics
The following is a list of comic book series based on the animated TV show The Simpsons and published by Bongo Comics in the United States.

 The first comic strips based on "The Simpsons" appeared in 1991 in the magazine "Simpsons Illustrated" (not to be confused with the comic publications from
2012 bearing the same name), which was a companion magazine to the show.

 The comic strips were popular and a one-shot comic book entitled "Simpsons Comics and Stories", containing three different stories, was released in 1993 for
the fans.

 The book was a success and due to this, the creator of "The Simpsons", Matt Groening, and his companions Bill Morrison, Mike Rote, Steve Vance and Cindy Vance
created the publishing company Bongo Comics.

 By the end of 1993, Bongo was publishing four titles: "Simpsons Comics", "Bartman", "Radioactive Man" and "Itchy & Scratchy Comics".
</item 6>

<item 7>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 7>

<item 8>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 8>

<item 9>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 9>

<item 10>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 10>
=== END ITEM id=dbc276807de5cc07 ===

=== ITEM id=be11a08f745429e0 ===
Gold sentences and facts:
Sentence 0: Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.
  Fact 0.0: Robin R. Bottin was born on April 1, 1959.
  Fact 0.1: Robin R. Bottin is an American special make-up effects creator.
Sentence 1:  Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing", with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".
  Fact 1.0: Robin R. Bottin is known for collaborations with directors John Carpenter, Paul Verhoeven and David Fincher.
  Fact 1.1: Robin R. Bottin worked with John Carpenter on “The Fog”.
  Fact 1.2: Robin R. Bottin worked with John Carpenter on “The Thing”.
  Fact 1.3: Robin R. Bottin worked with Paul Verhoeven on “RoboCop”.
  Fact 1.4: Robin R. Bottin worked with Paul Verhoeven on “Total Recall”.
  Fact 1.5: Robin R. Bottin worked with Paul Verhoeven on “Basic Instinct”.
  Fact 1.6: Robin R. Bottin worked with David Fincher on “Se7en”.
  Fact 1.7: Robin R. Bottin worked with David Fincher on “Fight Club”.
Sentence 2: David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.
  Fact 2.0: David Andrew Leo Fincher was born on August 28, 1962.
  Fact 2.1: David Andrew Leo Fincher is an American director and producer.
  Fact 2.2: David Andrew Leo Fincher is notably known for films, television series and music videos.
Sentence 3:  He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).
  Fact 3.0: David Fincher was nominated for the Academy Award for Best Director for “The Curious Case of Benjamin Button” (2008).
  Fact 3.1: David Fincher was nominated for the Academy Award for Best Director for “The Social Network” (2010).
Sentence 4:  For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
  Fact 4.0: For “The Social Network”, David Fincher won the Golden Globe Award for Best Director.
  Fact 4.1: For “The Social Network”, David Fincher won the BAFTA Award for Best Direction.

Retrieved items:
<item 1>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 1>

<item 2>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 4>

<item 5>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 5>

<item 6>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 6>

<item 7>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 7>

<item 8>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 8>

<item 9>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 9>

<item 10>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 10>
=== END ITEM id=be11a08f745429e0 ===

=== ITEM id=efd81e4a458e2250 ===
Gold sentences and facts:
Sentence 0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area women role models over 85 years old who remained artistically active.
  Fact 0.0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a documentary film.
  Fact 0.1: Still Kicking: Six Artistic Women of Project Arts & Longevity was released in 2006.
  Fact 0.2: Still Kicking: Six Artistic Women of Project Arts & Longevity runs 32 minutes.
  Fact 0.3: Still Kicking: Six Artistic Women of Project Arts & Longevity was made by Pacific Grove filmmaker Greg Young.
  Fact 0.4: Still Kicking: Six Artistic Women of Project Arts & Longevity features six Bay Area women role models.
  Fact 0.5: The six women featured in Still Kicking are over 85 years old.
  Fact 0.6: The six women featured in Still Kicking remained artistically active.
Sentence 1: Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
  Fact 1.0: Time Bombs is a film.
  Fact 1.1: Time Bombs is a 2008 film.
  Fact 1.2: Time Bombs is a Canadian film.
  Fact 1.3: Time Bombs was produced by Productions de la ruelle.

Retrieved items:
<item 1>
Still Kicking: The Fabulous Palm Springs Follies
Still Kicking: The Fabulous Palm Springs Follies is a 1997 American short documentary film directed by Mel Damski.

 It was nominated at the 70th Academy Awards for Best Documentary Short Subject.

 It features The Fabulous Palm Springs Follies, which was formerly staged at the Palm Springs, California Plaza Theatre.
</item 1>

<item 2>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
</item 2>

<item 3>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 3>

<item 4>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 4>

<item 5>
Time Bomb High School
Time Bomb High School is the second album by the Reigning Sound.

 It was released in 2002 on In the Red Records.

 The album featured the original Reigning Sound lineup of Greg Cartwright on lead vocals and guitar; Alex Greene on organ, piano, guitar, and backing vocals;
Jeremy Scott on bass, and backing vocals; and Greg Roberson on drums.

  Howlin' Pelle Almqvist of The Hives commented in "Rolling Stone" that Time Bomb High School was his favorite record of 2002.
</item 5>

<item 6>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 6>

<item 7>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 7>

<item 8>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 8>

<item 9>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 9>

<item 10>
Kick Over the Traces
Kick Over the Traces is a compilation album by the American punk rock band Pinhead Gunpowder.

 The album contains tracks from Pinhead Gunpowder releases since the band formed in 1990.

 The album was released through Recess Records on June 16, 2009.

 The Japanese edition features a second disc recorded live at 924 Gilman Street on February 10, 2008, and features different cover art.
</item 10>
=== END ITEM id=efd81e4a458e2250 ===

=== ITEM id=a265ad55e3b5fb69 ===
Gold sentences and facts:
Sentence 0: Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.
  Fact 0.0: Bumble and bumble Products, LLC is one of 27 brands.
  Fact 0.1: The 27 brands are owned by Estée Lauder Companies, Inc.
  Fact 0.2: Bumble and bumble Products, LLC is owned by Estée Lauder Companies, Inc.
Sentence 1:  The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown Manhattan, New York City.
  Fact 1.0: Estée Lauder Companies, Inc. owns a diverse portfolio of brands.
  Fact 1.1: The brand portfolio is distributed internationally through retail channels.
  Fact 1.2: The brand portfolio is distributed internationally through digital commerce.
  Fact 1.3: Estée Lauder Companies, Inc. has its headquarters in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 1>

<item 2>
A. H. Belo
A. H. Belo Corporation is a Dallas-based media company that owns newspapers in North Texas.

 The current corporation was formed when Belo Corporation separated its broadcasting and publishing operations into two corporations.

 A. H. Belo also owns a part interest in Classified Ventures.

 Its CEO is Robert Decherd who had headed Belo before the spin-off.

 The company has its headquarters in the Belo Building in Downtown Dallas.
</item 2>

<item 3>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 3>

<item 4>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 4>

<item 5>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 5>

<item 6>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 6>

<item 7>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 7>

<item 8>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 8>

<item 9>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 9>

<item 10>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 10>
=== END ITEM id=a265ad55e3b5fb69 ===

=== ITEM id=3703bbeb06199a5e ===
Gold sentences and facts:
Sentence 0:  His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files" (1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
  Fact 0.0: The subject of this sentence portrayed Bret Maverick in the series Maverick from 1957 to 1962.
  Fact 0.1: The subject of this sentence portrayed Bret Maverick in the series Bret Maverick from 1981 to 1982.
  Fact 0.2: The subject of this sentence portrayed Jim Rockford in the series The Rockford Files from 1974 to 1980.
  Fact 0.3: The subject of this sentence portrayed Jim Egan in the series 8 Simple Rules from 2003 to 2005.
Sentence 1: The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.
  Fact 1.0: The Notebook is a 2004 film.
  Fact 1.1: The Notebook is an American romantic drama film.
  Fact 1.2: The Notebook was directed by Nick Cassavetes.
  Fact 1.3: The Notebook is based on a 1996 novel.
  Fact 1.4: The 1996 novel has the same name as the film The Notebook.
  Fact 1.5: The 1996 novel was written by Nicholas Sparks.

Retrieved items:
<item 1>
The Country Girl (1954 film)
The Country Girl is a 1954 American drama film directed by George Seaton and starring Bing Crosby, Grace Kelly, and William Holden.

 Adapted by George Seaton from Clifford Odets' 1950 play of the same name, the film is about an alcoholic has-been actor struggling with the one last chance
he's been given to resurrect his career.

 Seaton won the Academy Award for Best Writing, Adapted Screenplay.

 It was entered in the 1955 Cannes Film Festival.
</item 1>

<item 2>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 2>

<item 3>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 3>

<item 4>
The New Maverick
The New Maverick is a 1978 made-for-TV movie based on the 1957 television series "Maverick", with James Garner as Bret Maverick, Charles Frank as newcomer
cousin Ben Maverick (son of Beau Maverick), Jack Kelly as Bart Maverick, and Susan Sullivan as Poker Alice Ivers.

 Garner had been 29 years old at the beginning of the original series and was 50 while filming "The New Maverick".

 The TV-movie was a pilot for the series "Young Maverick", which featured Frank and only lasted a few episodes.

 Directed by Hy Averback and written by Juanita Bartlett, the movie was filmed while Garner's series "The Rockford Files" was on hiatus.

 Garner would later star in "Bret Maverick", another attempt at a television series revival inspired by this TV-movie, for the 1981-82 season.
</item 4>

<item 5>
Enamorada (film)
Enamorada ("Enamoured") is a 1946 Mexican drama film directed by Emilio Fernández and starring María Félix and Pedro Armendáriz.

 The film was remade in 1950 as "The Torch" with Armendáriz repeating his role alongside Paulette Goddard who was credited as an associate producer on the film.
</item 5>

<item 6>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 6>

<item 7>
Mister Buddwing
Mister Buddwing is a 1966 American film drama directed by Delbert Mann and starring James Garner.
</item 7>

<item 8>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 8>

<item 9>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 9>

<item 10>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 10>
=== END ITEM id=3703bbeb06199a5e ===
