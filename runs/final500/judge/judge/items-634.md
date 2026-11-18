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

This file has 12 items.

=== ITEM id=7998e0d8ec9f0e3e ===
Gold sentences and facts:
Sentence 0:  Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any affiliation with the work.
  Fact 0.0: The film takes its title from an album by Leonard Cohen.
  Fact 0.1: Ira Nadel wrote a biography of Leonard Cohen.
  Fact 0.2: The subject of the film is not Leonard Cohen.
  Fact 0.3: The subject of the film has no affiliation with the work.
Sentence 1: Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and painter.
  Fact 1.0: Leonard Norman Cohen was born on September 21, 1934.
  Fact 1.1: Leonard Norman Cohen died on November 7, 2016.
  Fact 1.2: Leonard Norman Cohen was a Canadian singer.
  Fact 1.3: Leonard Norman Cohen was a Canadian songwriter.
  Fact 1.4: Leonard Norman Cohen was a Canadian musician.
  Fact 1.5: Leonard Norman Cohen was a Canadian poet.
  Fact 1.6: Leonard Norman Cohen was a Canadian novelist.
  Fact 1.7: Leonard Norman Cohen was a Canadian painter.

Retrieved items:
<item 1>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 1>

<item 2>
For Those Who Think Young (album)
For Those Who Think Young, appearing on the album cover as (for those who think young) and originally to be entitled for those who think jung, was the third
album by Canadian new wave band Rough Trade; it was released in 1981 (True North TN-48 in Canada; Boardwalk NB-33261-1, US; Big Time, UK; CBS 85385, The
Netherlands).

 It climbed to #9 in Canada on the "RPM" Top 50 Albums Chart on November 7, 1981 (putting it at #1 on the CANCON Chart listing), and held the position for three
weeks, dropping out of the Top 50 after sixteen weeks on February 6 of the following year.

 It was certified gold in Canada by the CRIA on November 1, 1981.

 The single "All Touch" gave the band its biggest commercial success, reaching #12 in Canada on the "RPM" Top 50 Singles Chart (#2 on the CANCON Chart) and #58
on the U.S. ""Billboard" Hot 100".
</item 2>

<item 3>
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 3>

<item 4>
Various Positions Tour
The Various Positions Tour was a concert tour by Leonard Cohen, in support of his album "Various Positions", released in 1984 (Canada) and 1985 in Europe and
the United States.
</item 4>

<item 5>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 5>

<item 6>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 6>

<item 7>
The Promised Land (Sagol 59 album)
The Promised Land: The Grateful Dead / Jerry Garcia Hebrew Project is a 2015 album by Sagol 59 and Ami Yares, featuring Hebrew language covers of songs by The
Grateful Dead and Jerry Garcia.

 It was released on April 17, 2015.

 The album was followed, in August 2015, by "The Promised Land Acoustic Sessions" EP, which contained four additional recordings: Hebrew renditions of "Ripple",
"Brown Eyed Women", "Friend of The Devil" and "Ship Of Fools".

 "The Pita Tapes" followed in March 2016, containing two unreleased demo recordings of songs often covered by The Grateful Dead and Garcia (Merle Haggard's
"Mama Tried" and Johnny Russell's "Catfish John").

 A nine-track limited edition Vinyl LP was released on April 2016.

 On August 1, 2016, studio recordings of "Deep Elem Blues" and "Half-Step Mississippi Uptown Toodeloo" were released digitally, to coincide with Garcia's
birthday.
</item 7>

<item 8>
Spiritual Unity (Marc Ribot album)
Spiritual Unity is a 2005 album by Marc Ribot released on Pi Recordings.

 The album features compositions by Albert Ayler who Ribot identifies as a significant influence.

 It was recorded at Orange Music Sound Studio, West Orange, New Jersey on October 28, 2004, except "Bells", which was recorded live at Tonic in New York City on
October 27, 2004.
</item 8>

<item 9>
The Cold Nose
The Cold Nose is the debut album by Department of Eagles.

 It has been reissued twice.

 It was originally released in the US as The Whitey On The Moon UK LP on Isota Records on October 21, 2003, and reissued in the UK as The Cold Nose on Melodic
Records on January 8, 2008.

 The American Dust October 7, 2008 US reissue contains four bonus tracks.

 The album is sample-heavy and multilayered.

 The second track, "Sailing By Night," contains a sample of the song "Meetings Along The Edge" from the Philip Glass and Ravi Shankar project "Passages", as
well as a sample from the Yellow Magic Orchestra track, Rydeen.

 The third track "Noam Chomsky Spring Break 2002" samples the Regina Spektor song "Prisoners".

 It also samples Astor Piazzolla's composition "Oblivion" in the opening and closing sequences.
</item 9>

<item 10>
The Burning Giraffe
In Dalí's own words his paintings form "a kind of allegory which serves to illustrate a certain insight, to follow the numerous narcissistic smells which ascend
from each of our drawers."
</item 10>
=== END ITEM id=7998e0d8ec9f0e3e ===

=== ITEM id=c5737d6b2ca9bc19 ===
Gold sentences and facts:
Sentence 0:  "Pinus" is the sole genus in the subfamily Pinoideae.
  Fact 0.0: Pinus is the sole genus in the subfamily Pinoideae.
Sentence 1:  The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35 unresolved species and many more synonyms.
  Fact 1.0: The Plant List was compiled by the Royal Botanic Gardens, Kew and the Missouri Botanical Garden.
  Fact 1.1: The Plant List accepts 126 species names of pines as current.
  Fact 1.2: The Plant List includes 35 unresolved species of pines.
  Fact 1.3: The Plant List includes many more synonyms.
Sentence 2:  It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
  Fact 2.0: The genus is sometimes considered to have only two species, B. monosperma and B. superba.
  Fact 2.1: The genus is sometimes expanded to include four or five species.

Retrieved items:
<item 1>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
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
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 3>

<item 4>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Pine
A pine is any conifer in the genus Pinus, , of the family Pinaceae.

 "Pinus" is the sole genus in the subfamily Pinoideae.

 The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35
unresolved species and many more synonyms.
</item 6>

<item 7>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 7>

<item 8>
Oak
An oak is a tree or shrub in the genus Quercus ( ; Latin "oak tree") of the beech family, Fagaceae.

 There are approximately 600 extant species of oaks.

 The common name "oak" also appears in the names of species in related genera, notably "Lithocarpus" (stone oaks), as well as in those of unrelated species such
as "Grevillea robusta" (silky oaks) and the Casuarinaceae (she-oaks).

 The genus "Quercus" is native to the Northern Hemisphere, and includes deciduous and evergreen species extending from cool temperate to tropical latitudes in
the Americas, Asia, Europe, and North Africa.

 North America contains the largest number of oak species, with approximately 90 occurring in the United States, while Mexico has 160 species of which 109 are
endemic.

 The second greatest center of oak diversity is China, which contains approximately 100 species.
</item 8>

<item 9>
Yellow pine
In ecology and forestry, yellow pine refers to a number of pine species which tend to grow in similar plant communities and yield similar strong wood.

 In the Western United States, yellow pine refers to Jeffrey pine or ponderosa pine.

 In the Southern United States, yellow pine refers to longleaf pine, shortleaf pine, slash pine, or Loblolly pine.

 In the United Kingdom, yellow pine refers to Eastern white pine or Scots pine.
</item 9>

<item 10>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 10>
=== END ITEM id=c5737d6b2ca9bc19 ===

=== ITEM id=93c66e025ad09301 ===
Gold sentences and facts:
Sentence 0:  The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
  Fact 0.0: The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck.
  Fact 0.1: Dave Brubeck was also one of the company's investors.
  Fact 0.2: The label is known more for its recordings of comedian Lenny Bruce.
  Fact 0.3: The label is known more for its recordings of jazz musician Vince Guaraldi.
  Fact 0.4: The label is known more for its recordings of the rock band Creedence Clearwater Revival.
Sentence 1: Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
  Fact 1.0: Vincent Anthony Guaraldi was born on July 17, 1928.
  Fact 1.1: Vincent Anthony Guaraldi died on February 6, 1976.
  Fact 1.2: Vincent Anthony Guaraldi was born Vincent Anthony Dellaglio.
  Fact 1.3: Vincent Anthony Guaraldi was an American jazz pianist.
  Fact 1.4: Vincent Anthony Guaraldi was noted for his innovative compositions and arrangements.
  Fact 1.5: Vincent Anthony Guaraldi was noted for composing music for animated television adaptations of the 'Peanuts' comic strip.
  Fact 1.6: Vincent Anthony Guaraldi performed on piano as a member of Cal Tjader's late 50s ensemble.
  Fact 1.7: Vincent Anthony Guaraldi had a solo career.
  Fact 1.8: Vincent Anthony Guaraldi's solo career included the radio hit 'Cast Your Fate to the Wind'.

Retrieved items:
<item 1>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 1>

<item 2>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 2>

<item 3>
Creedence Clearwater Revival (album)
Creedence Clearwater Revival is the debut studio album by the American rock band Creedence Clearwater Revival, released in 1968.
</item 3>

<item 4>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 4>

<item 5>
Fantasy Records
Fantasy Records is an American record company and label founded by brothers Max and Sol Weiss in 1949.

 The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known
more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
</item 5>

<item 6>
The Long Road Home
The Long Road Home: The Ultimate John Fogerty/Creedence Collection is a compilation album by American roots rock singer-songwriter John Fogerty, released on
November 1, 2005, by Fantasy Records.

 It compiles songs from Fogerty's solo career and his band Creedence Clearwater Revival.

 The title refers to Fogerty's return to Fantasy Records, after a lengthy stint with Warner Bros.

 Records and a brief stint with DreamWorks Records.
</item 6>

<item 7>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 7>

<item 8>
Jerry Beck
Peabody and Sherman", DreamWorks' "Madagascar", and "Bee Movie".

 Beck is also an entertainment industry consultant for TV and home entertainment productions and releases related to classic cartoons and operates the blog
"Cartoon Research."

 He appears frequently as a documentary subject and audio commentator on releases of A&E's "Cartoons Go to War" as well as DVD collections of "Looney Tunes",
"Popeye the Sailor", and "Woody Woodpecker" cartoons, on which he serves as consultant and curator.
</item 8>

<item 9>
The Best of Creedence Clearwater Revival
The Best of Creedence Clearwater Revival is a compilation album by American rock band Creedence Clearwater Revival, released in 1977.

 The album features all the tracks of Chronicle which was released a year earlier in 1976, as well as "Good Golly Miss Molly", "Born on the Bayou", "Cotton
Fields", "Hello Mary Lou", "The Midnight Special", "Walk on the Water", and "Night Time Is the Right Time" which would appear on the 1986 release "Chronicle,
Vol.

 2", and "Bootleg" which does not appear on either volume.
</item 9>

<item 10>
Vince Guaraldi
Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions
and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of
Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
</item 10>
=== END ITEM id=93c66e025ad09301 ===

=== ITEM id=c989e48a34ddc460 ===
Gold sentences and facts:
Sentence 0:  He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink, T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
  Fact 0.0: This producer has produced songs for Justin Bieber.
  Fact 0.1: This producer has produced songs for Monica.
  Fact 0.2: This producer has produced songs for Britney Spears.
  Fact 0.3: This producer has produced songs for Usher.
  Fact 0.4: This producer has produced songs for Keri Hilson.
  Fact 0.5: This producer has produced songs for T.I.
  Fact 0.6: This producer has produced songs for Nelly Furtado.
  Fact 0.7: This producer has produced songs for Kevin Cossom.
  Fact 0.8: This producer has produced songs for Ciara.
  Fact 0.9: This producer has produced songs for Mariah Carey.
  Fact 0.10: This producer has produced songs for Timbaland.
  Fact 0.11: This producer has produced songs for Madonna.
  Fact 0.12: This producer has produced songs for J. Cole.
  Fact 0.13: This producer has produced songs for Kanye West.
  Fact 0.14: This producer has produced songs for Whitney Houston.
  Fact 0.15: This producer has produced songs for Missy Elliott.
  Fact 0.16: This producer has produced songs for M.I.A.
  Fact 0.17: This producer has produced songs for Justin Timberlake.
  Fact 0.18: This producer has produced songs for JoJo.
  Fact 0.19: This producer has produced songs for Joe Jonas.
  Fact 0.20: This producer has produced songs for Simple Plan.
  Fact 0.21: This producer has produced songs for The Clutch.
  Fact 0.22: This producer has produced songs for Pink.
  Fact 0.23: This producer has produced songs for T-Pain.
  Fact 0.24: This producer has produced songs for Diddy.
  Fact 0.25: This producer has produced songs for Meek Mill.
  Fact 0.26: This producer has produced songs for Bjork.
  Fact 0.27: This producer has produced songs for Duran Duran.
Sentence 1:  Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.
  Fact 1.0: Cole was raised in Fayetteville, North Carolina.
  Fact 1.1: Cole initially gained recognition as a rapper.
  Fact 1.2: Cole initially gained recognition as a rapper after the release of his debut mixtape.
  Fact 1.3: Cole's debut mixtape is titled The Come Up.
  Fact 1.4: Cole's debut mixtape The Come Up was released in early 2007.

Retrieved items:
<item 1>
Lil Ru
Sylvester Samuels better known by his stage name Lil Ru, is an American rapper from Ridgeway, South Carolina currently signed to Def Jam Recordings.

 His debut album, 21 & Up was released on August 25, 2009.

 The Ridgeway native was 16 when he made his professional foray into the music business.

 Inspired by New Orleans’ innovative Cash Money Crew, Ru began making a name for himself on his local music scene, doing live shows and pressing up his own CDs.

 His hard-hitting lyrics and entrepreneurial spirit caught the attention of fellow South Carolinian Angie Stone.

 Shortly after the neo-soul songstress helped him secure a deal with Elektra Records, Ru found himself unsigned again, among the artists lost in the shuffle
after the label merged with Atlantic Records.

 Music fans first heard him on his 2001 debut single Will Destroy.

 He then released his 2002 follow up, Shawty What You Doin’.

 Both songs reached the Billboard R&B/Hip-Hop charts and helped land him at his next label home, Capitol Records.
</item 1>

<item 2>
J. Cole
Jermaine Lamarr Cole (born January 28, 1985), better known by his stage name J. Cole, is an American hip hop recording artist and record producer.

 Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.

 Intent on further pursuing a solo career as a rapper, he went on to release two additional mixtapes after signing to Jay Z's Roc Nation imprint in 2009.
</item 2>

<item 3>
Daniel Boone Native Gardens
Daniel Boone Native Gardens, located in Boone, North Carolina, United States, has a collection of North Carolina native plants in an informal landscaped design.

 The gardens are open daily from May to October.
</item 3>

<item 4>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 4>

<item 5>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
</item 5>

<item 6>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 6>

<item 7>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 7>

<item 8>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 8>

<item 9>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 9>

<item 10>
Zach Crowell
Zach Crowell is a Grammy Award-winning songwriter and record producer, who was born and raised in Nashville, Tennessee.

 Crowell is involved in a joint publishing deal with Warner/Chappell Music, Combustion Music and Taperoom Music.
</item 10>
=== END ITEM id=c989e48a34ddc460 ===

=== ITEM id=5d876db5fdc12d40 ===
Gold sentences and facts:
Sentence 0:  The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic Conference (MAAC).
  Fact 0.0: The Gaels were led by sixth year head coach Tim Cluess.
  Fact 0.1: The Gaels played their home games at the Hynes Athletic Center.
  Fact 0.2: The Gaels were members of the Metro Atlantic Athletic Conference (MAAC).
Sentence 1:  The Cyclones were coached by Steve Prohm, who was in his 1st season.
  Fact 1.0: The Cyclones were coached by Steve Prohm.
  Fact 1.1: Steve Prohm was in his 1st season.

Retrieved items:
<item 1>
2010–11 Iona Gaels men's basketball team
The 2010–11 Iona Gaels men's basketball team represented Iona College during the 2010–11 NCAA Division I men's basketball season.

 The Gaels, led by first year head coach Tim Cluess, played their home games at Hynes Athletics Center and are members of the Metro Atlantic Athletic
Conference.

 They finished the season 25–12, 13–5 in MAAC and lost in the championship game of the 2011 MAAC Men's Basketball Tournament to Saint Peter's.

 They were invited to the 2011 CollegeInsider.com Tournament where they advanced to the championship game before falling to Santa Clara.
</item 1>

<item 2>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 2>

<item 3>
2015–16 Iona Gaels men's basketball team
The 2015–16 Iona Gaels men's basketball team represented Iona College during the 2015–16 NCAA Division I men's basketball season.

 The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference (MAAC).

 They finished the season 22–11, 16–4 in MAAC play to finish in second place.

 They defeated Canisius, Siena, and Monmouth to be champions of the MAAC Tournament and earn the conference's automatic bid to the NCAA Tournament where, as a
#13 seed, they lost in the first round to Iowa State.
</item 3>

<item 4>
2014–15 Iona Gaels men's basketball team
The 2014–15 Iona Gaels men's basketball team represented Iona College during the 2014–15 NCAA Division I men's basketball season.

 The Gaels, led by fifth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 26–9, 17–3 in MAAC play to finish win the MAAC regular season championship.

 They defeated Siena and Monmouth to advance to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament where they
lost in the first round to Rhode Island.
</item 4>

<item 5>
2016–17 Iona Gaels men's basketball team
The 2016–17 Iona Gaels men's basketball team represented Iona College during the 2016–17 NCAA Division I men's basketball season.

 The Gaels, led by seventh year head coach Tim Cluess, played their home games at the Hynes Athletic Center in New Rochelle, New York as members of the Metro
Atlantic Athletic Conference (MAAC).

 They finished the season 22–13, 12–8 in MAAC play to finish in a tie for third place.

 They defeated Rider, Saint Peter's and Siena to be champions of the MAAC Tournament.

 They received the MAAC's automatic bid to the NCAA Tournament where they lost in the First Round to Oregon.
</item 5>

<item 6>
2011–12 Iona Gaels men's basketball team
The 2011–12 Iona Gaels men's basketball team represented Iona College during the 2011–12 NCAA Division I men's basketball season.

 The Gaels, led by second year head coach Tim Cluess, played their home games at the Hynes Athletic Center and are members of the Metro Atlantic Athletic
Conference.

 The Gaals were MAAC regular season champions but failed to win the MAAC Basketball Tournament after losing to Fairfield in the semifinals.

 They received an at–large bid to the 2012 NCAA Tournament where they lost to BYU in the "First Four" round.

 The Gaels led BYU by 25 points in the 2nd half.

 It was the largest comeback in NCAA Tournament history.
</item 6>

<item 7>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 7>

<item 8>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 8>

<item 9>
2013–14 Iona Gaels men's basketball team
The 2013–14 Iona Gaels men's basketball team represented Iona College during the 2013–14 NCAA Division I men's basketball season.

 The Gaels, led by fourth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 22–11, 17–3 to win the MAAC regular season championship.

 They advanced to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season conference champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament
where they lost in the first round to Louisiana Tech.
</item 9>

<item 10>
2012–13 Iona Gaels men's basketball team
The 2012–13 Iona Gaels men's basketball team represented Iona College during the 2012–13 NCAA Division I men's basketball season.

 The Gaels, led by third year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 20–14, 11–7 in MAAC play to finish in a tie for fourth place.

 They were champions of the MAAC Tournament, defeating Manhattan in the championship game, to earn an automatic bid to the 2013 NCAA Tournament where they lost
in the second round to Ohio State.
</item 10>
=== END ITEM id=5d876db5fdc12d40 ===

=== ITEM id=894f0d75647ebf8a ===
Gold sentences and facts:
Sentence 0: Pravda (Russian: Правда ; ] , "Truth") is a Russian broadsheet newspaper, formerly the official newspaper of the Communist Party of the Soviet Union, when it was one of the most influential papers in the country with a circulation of 11 million.
  Fact 0.0: Pravda's Russian name is Правда, which means 'Truth'.
  Fact 0.1: Pravda is a Russian broadsheet newspaper.
  Fact 0.2: Pravda was formerly the official newspaper of the Communist Party of the Soviet Union.
  Fact 0.3: When Pravda was the official newspaper of the Communist Party of the Soviet Union, it was one of the most influential papers in the country.
  Fact 0.4: When Pravda was one of the most influential papers in the country, its circulation was 11 million.
Sentence 1: Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.
  Fact 1.0: Stiffelio is an opera.
  Fact 1.1: Stiffelio is an opera in three acts.
  Fact 1.2: Stiffelio is by Giuseppe Verdi.
  Fact 1.3: Stiffelio uses an Italian libretto by Francesco Maria Piave.

Retrieved items:
<item 1>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 1>

<item 2>
Graph query row:
work: Stiffelio
kind: opera
genre: None
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 4>

<item 5>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 5>

<item 6>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 6>

<item 7>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 7>

<item 8>
List of major opera composers
This list provides a guide to opera composers, as determined by their presence on a majority of compiled lists of significant opera composers.

 (See the "Lists Consulted" section for full details.)

 The composers run from Jacopo Peri, who wrote the first ever opera in late 16th century Italy, to John Adams, one of the leading figures in the contemporary
operatic world.

 The brief accompanying notes offer an explanation as to why each composer has been considered major.

 Also included is a section about major women opera composers, compiled from the same lists.

 For an introduction to operatic history, see opera.

 The organisation of the list is by birthdate.
</item 8>

<item 9>
Graph query row:
work: Pravda
kind: Soviet newspaper
genre: None
</item 9>

<item 10>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 10>
=== END ITEM id=894f0d75647ebf8a ===

=== ITEM id=6b9a89f787ab1d6b ===
Gold sentences and facts:
Sentence 0: Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.
  Fact 0.0: Return to Never Land is also known as Peter Pan 2.
  Fact 0.1: Return to Never Land is also known as Peter Pan In: Return to Never Land.
  Fact 0.2: Return to Never Land is a 2002 American film.
  Fact 0.3: Return to Never Land is an animated musical fantasy-adventure film.
  Fact 0.4: Return to Never Land was produced by Walt Disney Television Animation.
  Fact 0.5: Return to Never Land was released by Walt Disney Pictures.
  Fact 0.6: Return to Never Land was distributed by Walt Disney Studios Motion Pictures.
Sentence 1:  The film was co-produced by Walt Disney Pictures and Mayhem Pictures.
  Fact 1.0: Return to Never Land was co-produced by Walt Disney Pictures.
  Fact 1.1: Return to Never Land was co-produced by Mayhem Pictures.

Retrieved items:
<item 1>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 1>

<item 2>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 2>

<item 3>
Peter Pan: Adventures in Never Land
Peter Pan: Adventures in Never Land (also known as Peter Pan in Disney's Return to Never Land) is a 2002 video game based on the Peter Pan Disney franchise.
</item 3>

<item 4>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 4>

<item 5>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 5>

<item 6>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 6>

<item 7>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 7>

<item 8>
Us (film)
Us is a 1991 television movie broadcast on CBS, produced, written and directed by Michael Landon.

 Landon also starred in the film, along with Barney Martin and Casey Peterson.

 It was a pilot for what would have been Landon's fourth consecutive television series; Landon's death that year precluded its going ahead.

 Landon played Jeff Hayes, a man just released from prison after serving many years due to being wrongfully convicted of killing a wealthy man's wife.
</item 8>

<item 9>
Never Land Books
The Never Land Books or Never Land Adventures are a series of short chapter books set in Never Land, the home of Peter Pan.

 They are based on the situations and characters established in the novel "Peter and the Starcatchers" and its sequels.

 Like the novels, they are written by Pulitzer Prize-winning humorist Dave Barry and suspense novelist Ridley Pearson, and illustrated by Greg Call.

 Although five books were planned, only three were published, in 2006–2008.

 The stories focus on supporting characters from the novels, such as the Indians, mermaids, pirates, and Lost Boys.

 They include:
</item 9>

<item 10>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 10>
=== END ITEM id=6b9a89f787ab1d6b ===

=== ITEM id=27bd422bde54be62 ===
Gold sentences and facts:
Sentence 0: Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.
  Fact 0.0: Paola Suárez was born on 23 June 1976.
  Fact 0.1: Paola Suárez is a retired professional tennis player.
  Fact 0.2: Paola Suárez is from Argentina.
Sentence 1:  She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.
  Fact 1.0: Paola Suárez was one of the most prominent women's doubles players throughout the early and mid-2000s.
  Fact 1.1: Paola Suárez won eight Grand Slam titles.
  Fact 1.2: All eight of Paola Suárez's Grand Slam titles were won with Virginia Ruano Pascual.
  Fact 1.3: Paola Suárez held the No. 1 doubles ranking for 87 non-consecutive weeks.
Sentence 2:  He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.
  Fact 2.0: Paola Suárez reached a career high singles ranking of 274 in July 2011.
  Fact 2.1: Paola Suárez is much better known as a leading doubles player with a huge serve.

Retrieved items:
<item 1>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 1>

<item 2>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 2>

<item 3>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 3>

<item 4>
Patricia Tarabini
Patricia Tarabini (born 6 August 1968) is an Argentine former tennis player.

 Tarabini represented her country and won the bronze Olympic medal at the 2004 Athens games, with Paola Suárez, where they lost 7–9 in the third set to eventual
gold medalists from China, Sun Tiantian and Li Ting.

 On May 9, 1988, Patricia reached her highest singles ranking; world number 29.

 Tarabini's highest doubles ranking was world number 12, which she achieved on August 17, 1998.

 Patricia turned pro in 1986, and won a total of 15 top-level doubles titles in her career.

 She is the 1996 French Open mixed doubles champion, which she won with Javier Frana.
</item 4>

<item 5>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 5>

<item 6>
Cleopatra Borel
Cleopatra Borel (from 2005 until 2010 Borel-Brown; born 10 March 1979) is a female shot putter from Plaisance, Mayaro, Trinidad and Tobago.

 Her personal best throw is 19.42 metres, achieved in July 2011 at the Paris Diamond League Meeting.

 She has a personal best of 19.48 metres on the indoor track, achieved in February 2004 in Blacksburg.
</item 6>

<item 7>
Paola Suárez
Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.

 She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano
Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.

 She was also a singles semifinalist at the 2004 French Open.
</item 7>

<item 8>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 8>

<item 9>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 9>

<item 10>
Clarisa Fernández
Clarisa Fernández (born 28 August 1981) is a retired Argentine tennis player who is best known for her semi-final appearance at the 2002 French Open, the first
Argentine since Sabatini in 1992 to reach that stage.

 Fernández was ranked 87th in the world at the time of her shocking result.

 She defeated, in order: (Q) Lubomira Bacheva, Jelena Kostanić, Kim Clijsters, (13) Elena Dementieva, Paola Suárez, before losing to second-seeded Venus
Williams in straight sets.
</item 10>
=== END ITEM id=27bd422bde54be62 ===

=== ITEM id=e1cc3c9b29cd2987 ===
Gold sentences and facts:
Sentence 0: Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.
  Fact 0.0: Neon City is also known as Anno 2053 in Italy.
  Fact 0.1: Neon City is also known as Neonski Grad in Serbia.
  Fact 0.2: Neon City is a 1991 Canadian film.
  Fact 0.3: Neon City is a post-apocalyptic science fiction film.
  Fact 0.4: Neon City was directed by Monte Markham.
  Fact 0.5: Neon City was written by Jeff Begun.
  Fact 0.6: Neon City was written by Ann Lewis Hamilton.
  Fact 0.7: Neon City's writing was credited under the pseudonym Buck Finch.
Sentence 1:  The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
  Fact 1.0: Michael Ironside is in the cast of Neon City.
  Fact 1.1: Vanity is in the cast of Neon City.
  Fact 1.2: Lyle Alzado is in the cast of Neon City.
  Fact 1.3: Valerie Wildman is in the cast of Neon City.
  Fact 1.4: Nick Klar is in the cast of Neon City.
  Fact 1.5: Juliet Landau is in the cast of Neon City.
  Fact 1.6: Richard Sanders is in the cast of Neon City.
  Fact 1.7: Monte Markham is in the cast of Neon City.
Sentence 2: Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and intimidating style of play.
  Fact 2.0: Lyle Martin Alzado was born on April 3, 1949.
  Fact 2.1: Lyle Martin Alzado died on May 14, 1992.
  Fact 2.2: Lyle Martin Alzado was a professional American football defensive end.
  Fact 2.3: Lyle Martin Alzado played in the National Football League.
  Fact 2.4: Lyle Martin Alzado was famous for his intense and intimidating style of play.

Retrieved items:
<item 1>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 1>

<item 2>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 2>

<item 3>
Graph query row:
creative_work: Neon City
alternate_titles: ['Anno 2053', 'Neonski Grad']
person: Lyle Martin Alzado
connection_to_work: None
related_person: Michael Ironside
related_person_connection_to_work: None
connection_between_people: None
</item 3>

<item 4>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 4>

<item 5>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 5>

<item 6>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 6>

<item 7>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 7>

<item 8>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 8>

<item 9>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 9>

<item 10>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 10>
=== END ITEM id=e1cc3c9b29cd2987 ===

=== ITEM id=c0b49d1e7d27491e ===
Gold sentences and facts:
Sentence 0: While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and Zoot Sims.
  Fact 0.0: The musician studied at the University of Michigan.
  Fact 0.1: While studying at the University of Michigan, the musician played with visiting musician Dizzy Gillespie.
  Fact 0.2: While studying at the University of Michigan, the musician played with visiting musician Miles Davis.
  Fact 0.3: While studying at the University of Michigan, the musician played with visiting musician Thad Jones.
  Fact 0.4: While studying at the University of Michigan, the musician played with visiting musician Billy Mitchell.
  Fact 0.5: After that, the musician played with Sonny Stitt, Count Basie and Al McKibbon.
  Fact 0.6: After that, the musician played with Cannonball Adderley, Percy Heath, Philly Joe Jones and Lou Donaldson.
  Fact 0.7: After that, the musician played with Donald Byrd, Kenny Dorham and Zoot Sims.
Sentence 1: Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.
  Fact 1.0: Miles Dewey Davis III was born on May 26, 1926.
  Fact 1.1: Miles Dewey Davis III died on September 28, 1991.
  Fact 1.2: Miles Dewey Davis III was an American jazz trumpeter.
  Fact 1.3: Miles Dewey Davis III was an American jazz bandleader.
  Fact 1.4: Miles Dewey Davis III was an American jazz composer.

Retrieved items:
<item 1>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 1>

<item 2>
Louis Smith (gymnast)
Louis Antoine Smith, MBE (born 22 April 1989) is a British artistic gymnast.

 He received a bronze medal and a silver medal on the pommel horse at the 2008 Beijing Olympics and the 2012 London Olympics respectively, with the former
marking the first time a British gymnast had placed in an Olympic event since 1928.

 He followed this up with a second consecutive silver medal on the pommel horse at the 2016 Rio Olympics, finishing behind teammate Max Whitlock.

 Smith was part of the Great Britain team that took the bronze in the men's artistic team all-around at the 2012 London Olympics.

 He is the only British gymnast to win Olympic medals in three separate Games.

 In 2015 he became the European champion on Pommel Horse.
</item 2>

<item 3>
Louis Smith (musician)
While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before
going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and
Zoot Sims.
</item 3>

<item 4>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 4>

<item 5>
Miles Davis
Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.

 He is among the most influential and acclaimed figures in the history of jazz and 20th century music.

 Davis adopted a variety of musical directions in his five-decade career which kept him at the forefront of a number of major stylistic developments in jazz.
</item 5>

<item 6>
Don Ellis
Donald Johnson Ellis (July 25, 1934 – December 17, 1978) was an American jazz trumpeter, drummer, composer, and bandleader.

 He is best known for his extensive musical experimentation, particularly in the area of time signatures. Later in his life he worked as a film composer,
contributing a score to 1971's "The French Connection" and 1973's "The Seven-Ups".
</item 6>

<item 7>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 7>

<item 8>
East Pole–West Pole divide
The East Pole–West Pole divide in the fields of cognitive psychology and cognitive neuroscience is an intellectual schism between researchers subscribing to the
nativist and empiricist schools of thought.

 The term arose from the fact that much of the theory and research supporting nativism, modularity of mind, and computational theory of mind originated at
several universities located on the East Coast, including Harvard University, the University of Michigan, Massachusetts Institute of Technology, and Tufts
University.

 Conversely, much of the research and theory supporting empiricism, emergentism, and embodied cognition originated at several universities located on the West
Coast, including the University of California, Berkeley, the Salk Institute, and, most notably, the University of California, San Diego.

 In reality, the divide is not so clear, with many universities and scholars on both coasts (as well as the Midwest and around the world) supporting each
position, as well as more moderate positions in between the two extremes.

 The phrase was coined by Jerry Fodor at an MIT conference on cognition, at which he referred to another researcher as a "West Coast theorist," apparently
unaware that the researcher worked at Yale University.
</item 8>

<item 9>
Thad Jones
Thaddeus Joseph Jones (March 28, 1923 – August 20, 1986) was an American jazz trumpeter, composer, and bandleader who has been called "one of the all-time
greatest jazz trumpet soloists."
</item 9>

<item 10>
Count Basie at Newport
Count Basie at Newport is a live album by jazz musician Count Basie and his orchestra.

 It was originally issued as Verve MGV 8243 and included only the tracks 1-8 and 13.

 Tracks 9-12 originally included in "Count Basie & Joe Williams/Dizzy Gillespie & Mary Lou Williams at Newport" (Verve MGV 8244).
</item 10>
=== END ITEM id=c0b49d1e7d27491e ===

=== ITEM id=aeceeed3ea83a336 ===
Gold sentences and facts:
Sentence 0: Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.
  Fact 0.0: Ferdi Taygan was born on December 5, 1956
  Fact 0.1: Ferdi Taygan is a former professional tennis player
  Fact 0.2: Ferdi Taygan is from the United States
Sentence 1: Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.
  Fact 1.0: Mahesh Shrinivas Bhupathi was born on 7 June 1974
  Fact 1.1: Mahesh Shrinivas Bhupathi is a retired Indian professional tennis player

Retrieved items:
<item 1>
1983 Donnay Indoor Championships – Doubles
Pavel Složil and Sherwood Stewart were the defending champions, but Složil did not participate this year.

 Stewart partnered Ferdi Taygan, losing in the first round.
</item 1>

<item 2>
Mahesh Bhupathi
Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.

 In 1997, he became the first Indian to win a Grand Slam tournament (with Rika Hiraki).

 With his win at the Australian Open mixed doubles in 2006, he joined the elite group of eight tennis players who have achieved a career Grand Slam in mixed
doubles.

 He is also the founder of International Premier Tennis League.

 In December 2016, Bhupathi was appointed as India's next non-playing Davis Cup captain and took over the reins from Anand Amritraj in February 2017.
</item 2>

<item 3>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 3>

<item 4>
1982 Volvo Masters – Doubles
Peter Fleming and John McEnroe won in the final 6–4, 6–3 against Sherwood Stewart and Ferdi Taygan.
</item 4>

<item 5>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 5>

<item 6>
Mahesh Bhupathi Tennis Academy
Mahesh Bhupathi Tennis Academy is a tennis training facility started by Indian tennis player Mahesh Bhupathi.

 The Academy has facilities across 9 states in India as well as in the UAE.
</item 6>

<item 7>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 7>

<item 8>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 8>

<item 9>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 9>

<item 10>
Ferdi Taygan
Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.

 He is of Turkish descent.
</item 10>
=== END ITEM id=aeceeed3ea83a336 ===

=== ITEM id=d04185f58a30c431 ===
Gold sentences and facts:
Sentence 0: Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal Navy surface vessels and submarines.
  Fact 0.0: Rosyth Dockyard is a large naval dockyard
  Fact 0.1: Rosyth Dockyard is on the Firth of Forth
  Fact 0.2: Rosyth Dockyard is at Rosyth, Fife, Scotland
  Fact 0.3: Rosyth Dockyard is owned by Babcock Marine
  Fact 0.4: Rosyth Dockyard formerly undertook refitting of Royal Navy surface vessels
  Fact 0.5: Rosyth Dockyard formerly undertook refitting of Royal Navy submarines
Sentence 1: The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.
  Fact 1.0: The Queen Elizabeth class is a class of two aircraft carriers
  Fact 1.1: The Queen Elizabeth class aircraft carriers belong to the United Kingdom's Royal Navy

Retrieved items:
<item 1>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 1>

<item 2>
Rosyth Dockyard
Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal
Navy surface vessels and submarines.

 Before its privatisation in the 1990s it was formally the Royal Naval Dockyard Rosyth.

 Its primary role is now as integration site for the Royal Navy's newest aircraft carriers - the "Queen Elizabeth"-class.
</item 2>

<item 3>
Graph query row:
entity: Queen Elizabeth-class aircraft carriers
relationship: PART_OF
organization: Royal Navy
</item 3>

<item 4>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 4>

<item 5>
Goliath (Rosyth)
Goliath is a crane in Rosyth Dockyard, Scotland, with a lift capacity of 1000 t , the largest in Britain.
</item 5>

<item 6>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 6>

<item 7>
Queen Elizabeth-class aircraft carrier
The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.

 The first,  , was named on 4 July 2014, with her ship commissioning planned for 2017, and an initial operating capability expected in 2018.

 The second, HMS "Prince of Wales" , is scheduled to be launched around Summer 2017, followed by commissioning in 2020 and service thereafter.

 On 5 September 2014, at the NATO 2014 Wales summit, the Prime Minister David Cameron announced that the second carrier would be brought into service, ending
years of uncertainty surrounding its future.

 This was confirmed by the November 2015 Government Strategic Defence Review, with both carriers entering service, one being available at any time.
</item 7>

<item 8>
HMS Prince of Wales (R09)
HMS "Prince of Wales" is the second "Queen Elizabeth"-class aircraft carrier under construction for the Royal Navy, with plans for active service from 2020.

 She is the seventh Royal Navy ship to have the name HMS "Prince of Wales".

 Construction of the ship began in 2011 at Rosyth Dockyard and in April 2016, was said to be around 80% structurally complete.
</item 8>

<item 9>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 9>

<item 10>
Rosyth railway station
Rosyth railway station serves the town of Rosyth in Fife, Scotland.

 The station is managed by Abellio ScotRail and lies on the Fife Circle Line, 23.6 km north of Edinburgh Waverley .

 It was opened in 1917 by the North British Railway (as "Rosyth Halt") to serve the nearby naval dockyard.
</item 10>
=== END ITEM id=d04185f58a30c431 ===
