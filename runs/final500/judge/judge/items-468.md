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

This file has 13 items.

=== ITEM id=dd3cdc1b93162714 ===
Gold sentences and facts:
Sentence 0: Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
  Fact 0.0: Ciaran Gribbin was born in 1976.
  Fact 0.1: Ciaran Gribbin is a Northern Irish singer-songwriter.
  Fact 0.2: Ciaran Gribbin is a guitarist.
  Fact 0.3: Ciaran Gribbin is a producer.
  Fact 0.4: Ciaran Gribbin hails from Castledawson, Northern Ireland.
Sentence 1: Brett Allen Scallions (born December 21, 1971) is an American musician.
  Fact 1.0: Brett Allen Scallions was born on December 21, 1971.
  Fact 1.1: Brett Allen Scallions is an American musician.
Sentence 2:  He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
  Fact 2.0: Brett Allen Scallions is the lead vocalist of the post-grunge band Fuel.
  Fact 2.1: Brett Allen Scallions is the rhythm guitarist of the post-grunge band Fuel.
  Fact 2.2: Brett Allen Scallions is the lyricist of the post-grunge band Fuel.

Retrieved items:
<item 1>
Puppet Strings
Puppet Strings is the fifth studio album by American rock band Fuel.

 Released March 4, 2014, it is their first studio effort since 2003's "Natural Selection" to feature original lead singer Brett Scallions and the first Fuel
album not to feature original songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 Additionally, no one who participated in the recording of Fuel's previous album "Angels & Devils" participated in the recording of "Puppet Strings".
</item 1>

<item 2>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 2>

<item 3>
The Best of Fuel
The Best of Fuel is a compilation album by American hard rock band Fuel.

 It contains songs from their first three full-length albums, "Sunburn", "Something Like Human", and "Natural Selection".

 All of the said albums featured vocalist Brett Scallions who departed from the group only a few months after the compilation's release.

 "The Best of Fuel" consists solely of the band's released singles, with the exception of "Million Miles" from "Natural Selection" (it features "Quarter" from
the same album instead).
</item 3>

<item 4>
Brett Scallions
Brett Allen Scallions (born December 21, 1971) is an American musician.

 He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
</item 4>

<item 5>
The entries profile rock, punk, and metal albums, bands, and musicians. Fuel’s *Puppet Strings* (2014) reunited singer Brett Scallions but featured none of the
prior album’s recording personnel. Jizzy Pearl sang on Quiet Riot’s *10* and L.A. Guns’ *Shrinking Violet*. Other subjects include J. Geils Band, Green Day side
projects and collaborators, Glassjaw and Daryl Palumbo, Clearview 77, Sons of Abraham,
</item 5>

<item 6>
Soul to Preach To
"Soul to Preach To" is the first single released from Fuel's album "Puppet Strings".

 It is also the first single released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 6>

<item 7>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 7>

<item 8>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 8>

<item 9>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 9>

<item 10>
Relient K’s “My Girlfriend,” written by 15-year-old Matt Thiessen, began as “Marilyn Manson Ate My Girlfriend”; it reflects a friend’s rejection of
Christianity. Thiessen also co-wrote Switchfoot’s “Rebuild” with Jon Foreman. Fuel’s 2007 *Angels & Devils* was Toryn Green’s only album with the band; 2014’s
*Puppet Strings* returned Brett Scallions but lacked original members Carl Bell and
</item 10>
=== END ITEM id=dd3cdc1b93162714 ===

=== ITEM id=ce0eeb718463805b ===
Gold sentences and facts:
Sentence 0: The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
  Fact 0.0: The Second Battle of Guam was fought from 21 July to 10 August 1944.
  Fact 0.1: The Second Battle of Guam was the American recapture of the Japanese-held island of Guam.
  Fact 0.2: Guam is a U.S. territory in the Mariana Islands.
  Fact 0.3: Japan captured Guam from the U.S. in the 1941 First Battle of Guam.
  Fact 0.4: The Second Battle of Guam took place during the Pacific campaign of World War II.
Sentence 1: The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and Antietam Creek as part of the Maryland Campaign.
  Fact 1.0: The Battle of Antietam was also known as the Battle of Sharpsburg, particularly in the South.
  Fact 1.1: The Battle of Antietam was fought on September 17, 1862.
  Fact 1.2: The Battle of Antietam was fought near Sharpsburg, Maryland.
  Fact 1.3: The Battle of Antietam was fought near Antietam Creek.
  Fact 1.4: The Battle of Antietam was part of the Maryland Campaign.

Retrieved items:
<item 1>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 1>

<item 2>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 2>

<item 3>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 3>

<item 4>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 4>

<item 5>
Maryland Campaign
The Maryland Campaign—or Antietam Campaign—occurred September 4–20, 1862, during the American Civil War.

 Confederate Gen. Robert E. Lee's first invasion of the North was repulsed by the Army of the Potomac under Maj. Gen. George B. McClellan, who moved to
intercept Lee and his Army of Northern Virginia and eventually attacked it near Sharpsburg, Maryland.

 The resulting Battle of Antietam was the bloodiest single-day battle in American history and is widely considered one of the major turning points of the war.
</item 5>

<item 6>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 6>

<item 7>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 7>

<item 8>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 8>

<item 9>
Battle of Antietam
The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and
Antietam Creek as part of the Maryland Campaign.

 It was the first field army–level engagement in the Eastern Theater of the American Civil War to take place on Union soil and is the bloodiest single-day
battle in American history, with a combined tally of 22,717 dead, wounded, or missing.
</item 9>

<item 10>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 10>
=== END ITEM id=ce0eeb718463805b ===

=== ITEM id=4b05cdbd2ed600b4 ===
Gold sentences and facts:
Sentence 0: Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.
  Fact 0.0: Colocasia is a genus of flowering plants.
  Fact 0.1: Colocasia is in the family Araceae.
  Fact 0.2: Colocasia is native to southeastern Asia.
  Fact 0.3: Colocasia is native to the Indian subcontinent.
Sentence 1: The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
  Fact 1.0: The genus Coronilla contains about 20 species.
  Fact 1.1: The Coronilla species are flowering plants.
  Fact 1.2: The Coronilla species are native to Europe.
  Fact 1.3: The Coronilla species are native to North Africa.

Retrieved items:
<item 1>
Colocasia
Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.

 Some species are widely cultivated and naturalized in other tropical and subtropical regions.

 Common names include tarul, karkala ko ganu, elephant-ear, taro, cocoyam, dasheen, chembu, champadhumpa, shavige gadde, and eddoe.

 Elephant-ear and cocoyam are also used for some other large-leaved genera in the Araceae, notably "Xanthosoma" and "Caladium".

 The generic name is derived from the ancient Greek word "kolokasion", which in the Greek botanist Dioscorides (1st century AD) meant the edible roots of both
"Colocasia esculenta" and "Nelumbo nucifera".

 It is thought that the edible roots of "Colocasia esculenta" have been cultivated in Asia for more than ten thousand years.

 The species "Colocasia esculenta" is an invasive species in wetlands along the American Gulf coast, where it threatens to displace native wetland plants.
</item 1>

<item 2>
The entries describe diverse flowering plants and groups, chiefly from Africa. Ice plants (Aizoaceae) include South African Aloinopsis and Conophytum, and
southern African Conicosia, whose wet-opening capsules release tiny seeds. Other genera span many families and regions: African irises and bulbous amaryllids;
Asian toad lilies and Rodgersia; European and North African Coronilla; and aquatic Alismataceae. Vascular plants are defined by xylem
</item 2>

<item 3>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 3>

<item 4>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 4>

<item 5>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 5>

<item 6>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 6>

<item 7>
The entries describe diverse plants and one crab genus. Mascarene Dictyosperma album, the wind-shedding “hurricane palm,” is widely cultivated but nearly
extinct in its native range. Canary Islands Adenocarpus viscosus grows above 1,800 m. Dregea vines occur in Africa and southern Asia; Duranta shrubs and trees
are ornamental, with D. erecta invasive in several regions. Tropical Dieffenbachia are popular houseplants
</item 7>

<item 8>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 8>

<item 9>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 9>

<item 10>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 10>
=== END ITEM id=4b05cdbd2ed600b4 ===

=== ITEM id=39ad4bc6f4ba1ba3 ===
Gold sentences and facts:
Sentence 0: Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.
  Fact 0.0: Isaac Asimov was born Isaak Ozimov.
  Fact 0.1: Isaac Asimov was born on January 2, 1920.
  Fact 0.2: Isaac Asimov's dates run from January 2, 1920 to April 6, 1992.
  Fact 0.3: Isaac Asimov was a Russian-American writer.
  Fact 0.4: Isaac Asimov was a professor of biochemistry at Boston University.
Sentence 1: Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.
  Fact 1.0: Michael Cunningham was born on November 6, 1952.
  Fact 1.1: Michael Cunningham is a U.S. novelist.
  Fact 1.2: Michael Cunningham is a U.S. screenwriter.

Retrieved items:
<item 1>
Isaac Asimov's Robots and Aliens
Isaac Asimov's Robots and Aliens is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 Each volume is complete in itself, but they form a continuing series.

 The series follows the action of the novels of the "Isaac Asimov's Robot City" series, with the same protagonists Derec and Ariel, and many other characters.

 The plot deals with the Three Laws and encounters between robots and different varieties of alien life.
</item 1>

<item 2>
The Last Question
"The Last Question" is a science fiction short story by American writer Isaac Asimov.

 It first appeared in the November 1956 issue of "Science Fiction Quarterly" and was anthologized in the collections "Nine Tomorrows" (1959), "The Best of Isaac
Asimov" (1973), "Robot Dreams" (1986), the retrospective "Opus 100" (1969), and in "Isaac Asimov: The Complete Stories, Vol.

 1" (1990).

 It was Asimov's favorite short story of his own authorship, and is one of a loosely connected series of stories concerning a fictional computer called
Multivac.

 The story overlaps science fiction, theology, and philosophy.
</item 2>

<item 3>
Isaac Asimov's Robot City: Odyssey
Isaac Asimov's Robot City: Odyssey is a young adult science fiction novel written in 1987 by Michael P. Kube-McDowell.

 It is part of the series Isaac Asimov's Robot City, inspired by Isaac Asimov's "Robot" series.

 The 1995 computer game Robot City is based on the plot of "Odyssey".
</item 3>

<item 4>
Isaac Asimov's Magical Worlds of Fantasy
Isaac Asimov's Magical Worlds of Fantasy is a series of twelve themed paperback fantasy and science fiction anthologies edited by Isaac Asimov, Martin H.
Greenberg and Charles G. Waugh, a companion set to the ten volume "Isaac Asimov's Wonderful Worlds of Science Fiction", produced by the same editors.

 It was published by Signet/New American Library from 1983 to 1991.

 Volumes 1 and 2 were also issued in hardcover in an omnibus collection titled "Isaac Asimov's Magical Worlds of Fantasy: Witches & Wizards".
</item 4>

<item 5>
Isaac Asimov's Robot City
Isaac Asimov's Robot City is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 It takes place between "The Robots of Dawn" and "Robots and Empire".

 Each volume is complete in itself, but they form a continuing series.

 The novels were written in response to a writing challenge issued by Asimov to write a series involving the Three Laws of Robotics, which brought about a
collaboration of several authors.

 Asimov provided outlines for stories which filled in the gap between Asimov's own robot stories and his "Foundation" series, explaining the disappearance of
the robots prior to the establishment of the galactic empire. "

Isaac Asimov's Robots and Aliens" followed in this series, with the same protagonists and many other characters.

 The common theme of all books of both series is the interaction between the characters and autonomous cities run and populated by robots (the "robot cities" of
the series title).

 Robot City was also released as a mystery game for the PC in 1995.

 The player takes the role of Derec.
</item 5>

<item 6>
Isaac Asimov's Robot City: Perihelion
Isaac Asimov's Robot City: Perihelion is a book written in 1988 by William F. Wu.

 It is part of the series Isaac Asimov's Robot City, which was inspired by Isaac Asimov's "Robot" series.
</item 6>

<item 7>
Michael Cunningham
Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.

 He is best known for his 1998 novel "The Hours", which won the Pulitzer Prize for Fiction and the PEN/Faulkner Award in 1999.

 Cunningham is a senior lecturer of creative writing at Yale University.
</item 7>

<item 8>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 8>

<item 9>
Wizards (Asimov anthology)
Wizards is an anthology of themed fantasy and science fiction short stories on the subject of wizards edited by Isaac Asimov, Martin H. Greenberg and Charles G.
Waugh.

 The first volume in their Isaac Asimov's Magical Worlds of Fantasy series, it was first published in paperback by Signet/New American Library in October 1983.

 It was later gathered together with "Witches", the second book in the series, into the omnibus hardcover collection "Isaac Asimov's Magical Worlds of Fantasy:
Witches & Wizards" (1985).
</item 9>

<item 10>
Isaac Asimov
Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.

 He was known for his works of science fiction and popular science.

 Asimov was a prolific writer, and wrote or edited more than 500 books and an estimated 90,000 letters and postcards. His books have been published in 9 of the
10 major categories of the Dewey Decimal Classification.
</item 10>
=== END ITEM id=39ad4bc6f4ba1ba3 ===

=== ITEM id=383d246ec76d66c9 ===
Gold sentences and facts:
Sentence 0: Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.
  Fact 0.0: Awolnation is an American alternative rock band.
  Fact 0.1: Awolnation was formed by Aaron Bruno.
  Fact 0.2: Awolnation is fronted by Aaron Bruno.
  Fact 0.3: Aaron Bruno was formerly of Under the Influence of Giants.
  Fact 0.4: Aaron Bruno was formerly of Home Town Hero.
  Fact 0.5: Aaron Bruno was formerly of Insurgence.
Sentence 1: This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.
  Fact 1.0: This Mortal Coil was a music collective.
  Fact 1.1: This Mortal Coil was led by Ivo Watts-Russell.
  Fact 1.2: Ivo Watts-Russell founded the British record label 4AD.

Retrieved items:
<item 1>
Awolnation
Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.
</item 1>

<item 2>
This Mortal Coil
This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.

 Although Watts-Russell and John Fryer were technically the only two official members, the band's recorded output featured a large rotating cast of supporting
artists, many of whom were otherwise associated with 4AD, including members of Cocteau Twins, Colourbox, and Dead Can Dance.

 The project became known for its gothic, dream pop sound, and released three full albums beginning in 1984 with "It'll End in Tears".
</item 2>

<item 3>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 3>

<item 4>
Hollow Moon (Bad Wolf)
"Hollow Moon (Bad Wolf)" is a song by American alternative rock band Awolnation.

 It was released as the lead single from their second studio album "Run" (2015).

 The song became their first number-one hit on the Billboard Alternative Songs chart in early 2015.
</item 4>

<item 5>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 5>

<item 6>
Peepshow (album)
Peepshow is the ninth studio album by English alternative rock band Siouxsie and the Banshees, released in September 1988 on Polydor.

 It was their first record as a quintet.

 With the arrival of multi-instrumentalist Martin McCarrick (known for his work as part of 4AD's This Mortal Coil), the group recorded a multifaceted album with
a variety of influences.

 Including the singles "Peek-a-Boo" and "The Last Beat of My Heart", the record was a commercial success, peaking at No. 68 on the "Billboard" 200 chart in the
week of 3 December 1988.

 It spent a total of 20 weeks on that chart.
</item 6>

<item 7>
Automatic (The Jesus and Mary Chain album)
Automatic is the third album by Scottish alternative rock band The Jesus and Mary Chain.

 The group on this record is basically the core duo of brothers William and Jim Reid with a drum machine providing percussion and even a synthesizer filling in
on bass guitar.

 The only other credited musician was Richard Thomas who joined the touring version of The Jesus and Mary Chain as a drummer.

 Thomas drummed on "Gimme Hell" and was a former member of Dif Juz.

 He also made appearances on Cocteau Twins' 1986 Victorialand LP and This Mortal Coil's 1986 Filigree & Shadow.
</item 7>

<item 8>
Mazzy Star
Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and
bassist Kendra Smith.

 Roback's friend Hope Sandoval became the group's vocalist when Smith left the band.
</item 8>

<item 9>
Awolnation discography
American alternative rock band Awolnation has released two studio albums, three extended plays, eight singles and ten music videos.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.

 The band's second studio album, "Run", was released on March 17, 2015.
</item 9>

<item 10>
Kill Your Heroes
"Kill Your Heroes" is a song by American alternative rock band Awolnation.

 It is written by lead singer Aaron Bruno and musician Brian West for the band's debut studio album "Megalithic Symphony", where it appears as the eighth track.

 "Kill Your Heroes" was released as the third and final single from "Megalithic Symphony" and reached the top 20 of the United States "Billboard" Alternative
Songs and Rock Songs charts.
</item 10>
=== END ITEM id=383d246ec76d66c9 ===

=== ITEM id=b9a2524ea6d60157 ===
Gold sentences and facts:
Sentence 0: The Australian women's national soccer team represents Australia in international women's soccer at the senior level.
  Fact 0.0: The Australian women's national soccer team represents Australia in international women's soccer.
  Fact 0.1: The Australian women's national soccer team represents Australia at the senior level.
Sentence 1:  The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.
  Fact 1.0: The Australian women's national soccer team's official nickname is the Matildas.
  Fact 1.1: The nickname Matildas comes from the Australian folk song "Waltzing Matilda".
  Fact 1.2: The Australian women's national soccer team was known as the Female Socceroos before 1995.
Sentence 2: "Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
  Fact 2.0: "Waltzing Matilda" is Australia's best-known bush ballad.
  Fact 2.1: "Waltzing Matilda" has been described as the country's "unofficial national anthem."

Retrieved items:
<item 1>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 1>

<item 2>
Vanuatu national under-23 football team
The Vanuatu national under-23 football team, also known as Vanuatu Cyclone, represents Vanuatu at U23 tournaments.

 The team is considered to be the feeder team for the Vanuatu national football team.

 They are controlled by the Vanuatu Football Federation.

 The team has gained notoriety for thrashing Micronesia's side 46–0 in the 2015 Pacific Games.
</item 2>

<item 3>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 3>

<item 4>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 4>

<item 5>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 5>

<item 6>
Spencer Park, Queensland
Spencer Park (currently known as Corporate Travel Management Stadium) is a football stadium located in the suburb of Newmarket, north of Brisbane, Queensland,
Australia.

 It is home to National Premier Leagues Queensland team Brisbane City FC.

 The stadium, while officially holding 10,000, seats only 3,000 under the grandstand, the "Gino Merlo Stand".

 The stadium opened in 1963 on the site of an old Brisbane City Council waste disposal site, and underwent a major renovation in 1981 with the construction of
the Gino Merlo Stand.

 The stadium is home to La Rustica restaurant, where traditional Italian foods, including pizza, are served.
</item 6>

<item 7>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 7>

<item 8>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 8>

<item 9>
Australia (Gyroscope song)
"Australia" is a song by Australian rock band Gyroscope, from the album "Breed Obsession".

 It was released as the third single from their highly successful album, which peaked at number one on the Australian Albums Chart.

 The song was released to radio in May 2008, and was released as a CD single and digital download on 14 June 2008, along with two acoustic pieces.

 One of those being a live acoustic version of "Australia", the other a live acoustic cover of Chris Isaak's "Wicked Game".
</item 9>

<item 10>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 10>
=== END ITEM id=b9a2524ea6d60157 ===

=== ITEM id=3203f6fe2f9c33c3 ===
Gold sentences and facts:
Sentence 0: Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
  Fact 0.0: Tímea Babos was born on 10 May 1993.
  Fact 0.1: Tímea Babos is a Hungarian tennis player.
Sentence 1: Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis Professionals (ATP).
  Fact 1.0: Jack Sock was born on September 24, 1992.
  Fact 1.1: Jack Sock is an American professional tennis player.
  Fact 1.2: Jack Sock is currently ranked world No. 21 in men's singles.
  Fact 1.3: The ranking is by the Association of Tennis Professionals (ATP).

Retrieved items:
<item 1>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 1>

<item 2>
2017 Apia International Sydney – Women's Doubles
Martina Hingis and Sania Mirza were the defending champions, but chose not to participate together this year.

 Hingis played alongside Coco Vandeweghe, but lost in the quarterfinals to Tímea Babos and Anastasia Pavlyuchenkova.

 Mirza teamed up with Barbora Strýcová, but lost in the final to Babos and Pavlyuchenkova, 4–6, 4–6.
</item 2>

<item 3>
Tímea Babos
Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
</item 3>

<item 4>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 4>

<item 5>
2016 Dubai Tennis Championships – Women's Doubles
Tímea Babos and Kristina Mladenovic were the defending champions, but chose not to compete together.

 Babos played alongside Julia Görges, but lost in the semifinals to Caroline Garcia and Mladenovic.

<br>
</item 5>

<item 6>
2013 Aegon Classic – Doubles
Tímea Babos and Hsieh Su-wei were the defending champions, but Babos decided not to participate.

<br>
</item 6>

<item 7>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 7>

<item 8>
2014 Suzhou Ladies Open – Doubles
Tímea Babos and Michaëlla Krajicek were the defending champions, having won the event in 2013, however both players chose not to participate.
</item 8>

<item 9>
Jack Sock
Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis
Professionals (ATP).

 He is one of the top ranked American men in both singles and doubles on the ATP Tour.

 A former junior US Open champion, Sock's singles success is highlighted by 7 ATP finals, including three titles.

 His career best result came at the 2017 Indian Wells Masters, where he reached the semi-finals.
</item 9>

<item 10>
2015 Malaysian Open – Doubles
Tímea Babos and Chan Hao-ching were the defending champions, but Chan chose not to participate this year.

 Babos chose to play in Monterrey, but lost in the first round.
</item 10>
=== END ITEM id=3203f6fe2f9c33c3 ===

=== ITEM id=d7d111e2b026e188 ===
Gold sentences and facts:
Sentence 0: 3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.
  Fact 0.0: 3:10 to Yuma is a 2007 American Western film.
  Fact 0.1: 3:10 to Yuma was directed by James Mangold.
  Fact 0.2: 3:10 to Yuma was produced by Cathy Konrad.
  Fact 0.3: Russell Crowe and Christian Bale star in the lead roles of 3:10 to Yuma.
  Fact 0.4: Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman have supporting performances in 3:10 to Yuma.
Sentence 1: Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and Wat in "A Knight's Tale".
  Fact 1.0: Alan Wray Tudyk was born on March 16, 1971.
  Fact 1.1: Alan Tudyk is an American actor.
  Fact 1.2: Alan Tudyk is an American voice actor.
  Fact 1.3: Alan Tudyk is known for playing Hoban "Wash" Washburne in the space western television series "Firefly" and in the movie "Serenity".
  Fact 1.4: Alan Tudyk is known for playing Alpha in the science fiction TV series "Dollhouse".
  Fact 1.5: Alan Tudyk is known for playing Tucker McGee in "Tucker & Dale vs. Evil".
  Fact 1.6: Alan Tudyk is known for playing Steve the Pirate in "" (the title is missing from the source sentence).
  Fact 1.7: Alan Tudyk is known for playing Wat in "A Knight's Tale".

Retrieved items:
<item 1>
Deadfall (2012 film)
Deadfall is a 2012 American crime drama film directed by Stefan Ruzowitzky, written by Zach Dean and starring Eric Bana, Olivia Wilde, and Charlie Hunnam.
</item 1>

<item 2>
3:10 to Yuma (2007 film)
3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead
roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.

 It is about a drought-impoverished rancher (Bale) who takes on the dangerous job of taking a notorious outlaw (Crowe) to justice.

 It is a remake of the , making it the second adaptation of Elmore Leonard's short story "Three-Ten to Yuma".

 Filming took place in various locations in New Mexico.

 "3:10 to Yuma" opened September 7, 2007, in the United States and received positive reviews from critics.
</item 2>

<item 3>
The Appaloosa
The Appaloosa (also known as Southwest to Sonora) is a 1966 American Western film Technicolor (set in the 1870s) from Universal Pictures starring Marlon Brando,
Anjanette Comer and John Saxon, who was nominated for a Golden Globe for Best Supporting Actor for his portrayal of a Mexican bandit.

 The film was directed by Sidney J. Furie, and shot in Wrightwood, Antelope Valley, and Lake Los Angeles, California, St. George, Utah, and Colorado City,
Arizona.
</item 3>

<item 4>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 4>

<item 5>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 5>

<item 6>
Gene Hackman
Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.

 In a career spanning nearly five decades, Hackman was nominated for five Academy Awards, winning Best Actor in "The French Connection" and Best Supporting
Actor in "Unforgiven".

 He won three Golden Globes and two BAFTAs.
</item 6>

<item 7>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 7>

<item 8>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 8>

<item 9>
Alan Tudyk
Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television
series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and
Wat in "A Knight's Tale".

 He also co-starred on the ABC sitcom "Suburgatory".

 Tudyk had a supporting role in the film "42" as baseball manager Ben Chapman and voiced roles in every Walt Disney Animation Studios film to date starting with
"Wreck-It Ralph".

 He has also featured in the web series "Con Man" and the films "" and "Rogue One: A Star Wars Story".
</item 9>

<item 10>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 10>
=== END ITEM id=d7d111e2b026e188 ===

=== ITEM id=aca24c03c4bbfa13 ===
Gold sentences and facts:
Sentence 0: Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
  Fact 0.0: Leitneria floridana is also called corkwood.
  Fact 0.1: Leitneria floridana is the sole species in the genus Leitneria.
  Fact 0.2: Leitneria floridana is a deciduous dioecious shrub or small tree.
  Fact 0.3: Leitneria floridana is found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
Sentence 1: Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.
  Fact 1.0: Tricyrtis is a genus of Asian flowering plants.
  Fact 1.1: Tricyrtis is in the lily family.
  Fact 1.2: Tricyrtis has approximately 20 known species.

Retrieved items:
<item 1>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 1>

<item 2>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 4>

<item 5>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 5>

<item 6>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 6>

<item 7>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 7>

<item 8>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 10>
=== END ITEM id=aca24c03c4bbfa13 ===

=== ITEM id=96e5e68924830d97 ===
Gold sentences and facts:
Sentence 0:  He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.
  Fact 0.0: He is mainly known for his spells at Real Sociedad.
  Fact 0.1: His offensive partnership with Nihat Kahveci at Real Sociedad was one of the best in Spain.
Sentence 1: Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
  Fact 1.0: Nihat Kahveci was born on 23 November 1979.
  Fact 1.1: Nihat Kahveci is a Turkish former footballer.
  Fact 1.2: Nihat Kahveci played as a forward.

Retrieved items:
<item 1>
Lippo Hertzka
Lippo Hertzka (19 November 1904 – 14 March 1951) was a Hungarian football player and manager.

 He played for Essener Turnerbund, MTK Budapest and Real Sociedad.

 After retiring, he coached seven teams, including Real Sociedad and Real Madrid, a team which he coached for 2 years (1930–1932) and led to an undefeated La
Liga championship during the 1931-32 season, which meant the first La Liga title for the "white" squad.

 He also won two league titles in Portugal for Benfica.
</item 1>

<item 2>
Real Sociedad
Real Sociedad de Fútbol, S.A.D., more commonly referred to as Real Sociedad (] ; "Royal Society") or La Real, is a Spanish football club based in the city of
San Sebastián, Basque Country, founded on 7 September 1909.

 It plays its home matches at the 32,000-capacity Anoeta Stadium.

 Real Sociedad won the Liga title in 1980–81 and 1981–82, and last finished runners-up in 2002–03.

 The club has also won the Copa del Rey twice, in 1909 and 1987.

 It contests the Basque derby against rivals Athletic Bilbao.

 Real Sociedad were founder members of La Liga in 1928, and its longest spell in the top flight was for 40 seasons, from 1967 to 2007.
</item 2>

<item 3>
2013–14 Real Sociedad season
The 2013–14 season was Real Sociedad's 67th season in La Liga.

 Real Sociedad finished 7th in the league and reached the semifinals of the Copa del Rey.

 The Basque failed to make it out of the group stages of the UEFA Champions League.
</item 3>

<item 4>
Real Sociedad Femenino
Real Sociedad Femenino is the women's football section of Real Sociedad de Fútbol.

 Founded in 2004
</item 4>

<item 5>
Darko Kovačević
Darko Kovačević (] ; Serbian Cyrillic: ; born 18 November 1973) is a Serbian former footballer who played as a forward.

 He began his career in Serbia with Proleter Zrenjanin and subsequently played for Red Star Belgrade, with whom he won a Yugoslav League title and two Yugoslav
Cups.

 His prolific performances earned him a move to Premier League side Sheffield Wednesday, although his time in England was less successful.

 He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.

 Kovačević also had positive spells with Italian club Juventus and Greek side Olympiacos.

 At international level, he represented Yugoslavia at the 1998 FIFA World Cup and at UEFA Euro 2000.
</item 5>

<item 6>
Ivan Toplak
Ivan Toplak (Serbian Cyrillic: Ивaн Toплaк; born September 21, 1931 in Belgrade) is a former Yugoslavian football player and manager.

 He earned 1 cap for Yugoslavia.

 He started his career at Branik in Maribor, where his family moved in 1943 from Belgrade.
</item 6>

<item 7>
C.D. Real Sociedad
Club Deportivo Real Sociedad, commonly known as Real Sociedad (] ), is a Honduran football club based on Tocoa, Colón, Honduras.
</item 7>

<item 8>
Sport Mundi Tournament
The Sport Mundi Tournament is a Spanish pre-season women's football invitational charitable tournament held every August in Irun since 2005.

 It is contested by four teams, usually including regional powerhouses Athletic Bilbao and Real Sociedad.

 Athletic is the most successful team in the competition with four titles, followed by Levante UD and Real Sociedad with two each.
</item 8>

<item 9>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 9>

<item 10>
Nihat Kahveci
Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
</item 10>
=== END ITEM id=96e5e68924830d97 ===

=== ITEM id=725e56e5b0e5ea72 ===
Gold sentences and facts:
Sentence 0: The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.
  Fact 0.0: The 2012 United States Libertarian National Convention was a convention of the Libertarian Party (LP).
  Fact 0.1: Delegates of the Libertarian Party chose the party's nominees at the 2012 United States Libertarian National Convention.
  Fact 0.2: The nominees chosen at the 2012 United States Libertarian National Convention were for President of the United States.
  Fact 0.3: The nominees chosen at the 2012 United States Libertarian National Convention were for Vice President of the United States.
  Fact 0.4: The nominees were chosen for the 2012 general election.
  Fact 0.5: The 2012 United States Libertarian National Convention was held May 2-6, 2012.
  Fact 0.6: The 2012 United States Libertarian National Convention was held in Las Vegas, Nevada.
  Fact 0.7: The 2012 United States Libertarian National Convention was held at the Red Rock Resort Spa and Casino.
Sentence 1:  Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.
  Fact 1.0: Gary Johnson is a former Governor of New Mexico.
  Fact 1.1: Gary Johnson won the presidential nomination.
  Fact 1.2: Gary Johnson won the presidential nomination on the first ballot.
Sentence 2: Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.
  Fact 2.0: Gary Earl Johnson was born on January 1, 1953.
  Fact 2.1: Gary Earl Johnson is an American businessman.
  Fact 2.2: Gary Earl Johnson is an American author.
  Fact 2.3: Gary Earl Johnson is an American politician.
Sentence 3:  He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.
  Fact 3.0: Gary Johnson was the 29th Governor of New Mexico.
  Fact 3.1: Gary Johnson was Governor of New Mexico from 1995 to 2003.
  Fact 3.2: Gary Johnson was a member of the Republican Party as Governor of New Mexico.

Retrieved items:
<item 1>
2012 Libertarian National Convention
The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United
States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.

 Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.

 Retired California state court judge Jim Gray won the vice-presidential nomination, also on the first ballot.

 The convention also chose to replace most of the Libertarian National Committee party officers and members-at-large.
</item 1>

<item 2>
Libertarian National Committee
The Libertarian National Committee (LNC) controls and manages the affairs, properties, and funds of the United States Libertarian Party.

 It is composed of the party officers, five at-large representatives elected every two years at the national convention, and a theoretical maximum of 10
regional representatives.

 The current chair, elected in June 2014, is Nicholas Sarwark.

 The Executive Director is Wes Benedict.
</item 2>

<item 3>
Gary Johnson presidential campaign, 2012
The 2012 presidential campaign of Gary Johnson, the 29th Governor of New Mexico, was announced on April 21, 2011.

 He declared his candidacy for the 2012 Republican Party nomination for President of the United States.

 On December 28, 2011, Johnson withdrew his candidacy for the Republican nomination, and declared his candidacy for the 2012 presidential nomination of the
Libertarian Party.

 The 2012 Libertarian National Convention was held during the first weekend of May 2012.

 On May 5, 2012, after promoting his libertarian-oriented political positions to delegates, Johnson received the most votes at the convention and became the
official 2012 Libertarian presidential nominee.

 On November 6, 2012, Johnson received just under 1% of the popular vote in the general election, amounting to more than 1.2 million votes, more than double
what the Barr/Root ticket received in 2008.

 This was the most successful result for a third-party presidential candidacy since 2000, and the best in the Libertarian Party's history by vote number at the
time.

 (Johnson ran again in 2016 and received nearly four times his 2012 vote total.)
</item 3>

<item 4>
Libertarian National Convention
The Libertarian National Convention is held every two years by the Libertarian Party (United States) to choose members of the Libertarian National Committee
(LNC), and to conduct other party business.

 In presidential election years, the convention delegates enact a platform and nominate the Libertarian presidential and vice-presidential candidates who then
face the nominees of other parties in the November general election.
</item 4>

<item 5>
Libertarian Party presidential primaries, 2016
The 2016 Libertarian Party presidential primaries and caucuses allowed electors to indicate non-binding preferences for the Libertarian Party's presidential
candidate.

 These differed from the Republican or Democratic presidential primaries and caucuses in that they did not appoint delegates to represent a candidate at the
party's convention to select the party's nominee for the United States presidential election.

 The party's nominee for the 2016 presidential election was chosen directly by registered delegates at the 2016 Libertarian National Convention, which ran from
May 26 to 30, 2016.

 The delegates nominated former New Mexico Governor Gary Johnson and former Massachusetts Governor Bill Weld.
</item 5>

<item 6>
David Bergland was the Libertarian presidential nominee in 1984. At the 2012 convention in Las Vegas, delegates nominated Gary Johnson and Jim Gray on the first
ballot and replaced most party officers. Johnson, a former New Mexico governor, switched from the Republican race to the Libertarian Party in 2011; he won over
1.2 million votes in 2012 and was nominated again in 2016 alongside Bill Weld. The 2016 primaries were nonbinding; convention delegates selected nominees
</item 6>

<item 7>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 7>

<item 8>
David Bergland
David Peter Bergland (born June 4, 1935) is an American politician who was the United States Libertarian Party's nominee at the 1983 Libertarian National
Convention for President of the United States in the 1984 presidential election.
</item 8>

<item 9>
Gary Johnson
Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.

 He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.

 He was also the Libertarian Party nominee for President of the United States in the 2012 and 2016 elections.
</item 9>

<item 10>
Political positions of Gary Johnson
Gary Johnson was the governor of New Mexico from 1995 to 2003 and ran for president in 2012 and 2016.

 In December 2011 he announced he would pursue the presidential nomination of the Libertarian Party for the 2012 general election.

 The Libertarian National Convention in May, 2012 chose Johnson as the party's candidate.

 In November 2014, Johnson announced he would pursue the presidential nomination of the Libertarian Party for the 2016 general election.

 Johnson has taken positions on many political issues as revealed through his public comments, his gubernatorial record, his Our America Initiative and his
campaigns to win the Republican and Libertarian nominations.
</item 10>
=== END ITEM id=725e56e5b0e5ea72 ===

=== ITEM id=779915c6f9e680d1 ===
Gold sentences and facts:
Sentence 0: Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on science and technology subjects.
  Fact 0.0: Popular Science is also known as PopSci.
  Fact 0.1: Popular Science is an American bi-monthly magazine.
  Fact 0.2: Popular Science carries popular science content.
  Fact 0.3: Popular science content refers to articles for the general reader on science and technology subjects.
Sentence 1: Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.
  Fact 1.0: Girlfriends was a women's magazine.
  Fact 1.1: Girlfriends provided critical coverage of culture.
  Fact 1.2: Girlfriends provided critical coverage of entertainment.
  Fact 1.3: Girlfriends provided critical coverage of world events.
  Fact 1.4: Girlfriends' coverage was from a lesbian perspective.
Sentence 2:  Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.
  Fact 2.0: Girlfriends was published monthly from San Francisco since 1993.
  Fact 2.1: Girlfriends was distributed nationwide by Disticor.
Sentence 3:  "Girlfriends" magazine ceased publication in 2006.
  Fact 3.0: Girlfriends magazine ceased publication in 2006.

Retrieved items:
<item 1>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 1>

<item 2>
The entries cover popular-science and technology publications, media organizations, and a computer maker. Austrian German-language Universum is financially
supported by the Austrian National Science Foundation; U.S. Popular Science, founded in 1872, is award-winning and internationally distributed. Popular science
explains science for general audiences across media. NASA-sponsored Astrobiology Magazine has reported on space missions and life-universe research since 2000.
Other titles include culture magazine PAPER, PCMag, Windows: The Official Magazine, independent
</item 2>

<item 3>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 3>

<item 4>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 4>

<item 5>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 5>

<item 6>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 6>

<item 7>
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 7>

<item 8>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 8>

<item 9>
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 9>

<item 10>
Popular science
Popular science (also pop-science or popsci) is interpretation of science intended for a general audience.

 While science journalism focuses on recent scientific developments, popular science is more broad-ranging.

 It may be written by professional science journalists or by scientists themselves.

 It is presented in many forms, including books, film and television documentaries, magazine articles, and web pages.
</item 10>
=== END ITEM id=779915c6f9e680d1 ===

=== ITEM id=09ec68c2957a9e81 ===
Gold sentences and facts:
Sentence 0:  He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".
  Fact 0.0: The person described has written a column for Fortune magazine for more than twenty years.
  Fact 0.1: The person described spent a decade at Esquire before writing for Fortune.
Sentence 1:  Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
  Fact 1.0: The described publication was founded in 1933.
  Fact 1.1: The described publication flourished during the Great Depression.
  Fact 1.2: The described publication flourished under the guidance of founders Arnold Gingrich, David A. Smart, and Henry L. Jackson.

Retrieved items:
<item 1>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 1>

<item 2>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 2>

<item 3>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 3>

<item 4>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 4>

<item 5>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 5>

<item 6>
This Is Bing Crosby
This Is Bing Crosby was a fifteen-minute five times a week daytime radio program featuring Bing Crosby acting as a disc jockey.

 Minute Maid quick frozen concentrated orange juice was promoted on the shows.
</item 6>

<item 7>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 7>

<item 8>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 8>

<item 9>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 9>

<item 10>
Dick Gordon (sports writer)
Charles Richards Gordon, known as Dick "Scoop" Gordon (January 15, 1911 – December 8, 2008), was an American sports journalist whose works were a regular
feature in venerable sports magazines like "The Sporting News", "Sports Illustrated", and "Baseball Digest".

 After earning his nickname "Scoop" in 1930 by reporting for "The Daily Princetonian" that golfing legend Bobby Jones would be retiring from active competition,
Gordon went on to a sports reporting career which ended in 2008.
</item 10>
=== END ITEM id=09ec68c2957a9e81 ===
