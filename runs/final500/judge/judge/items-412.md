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

=== ITEM id=b8e39684419a0363 ===
Gold sentences and facts:
Sentence 0: Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
  Fact 0.0: Leycesteria is a genus of flowering plants.
  Fact 0.1: Leycesteria is in the honeysuckle family Caprifoliaceae.
  Fact 0.2: Leycesteria is native to temperate Asia in the Himalaya.
  Fact 0.3: Leycesteria is native to southwestern China.
Sentence 1: Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.
  Fact 1.0: Anigozanthos is a small genus of Australian plants.
  Fact 1.1: Anigozanthos is in the Bloodwort family Haemodoraceae.

Retrieved items:
<item 1>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 1>

<item 2>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 2>

<item 3>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 3>

<item 4>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 4>

<item 5>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 5>

<item 6>
Forests of Australia
Australia has many forests of importance due to significant features, despite being one of the driest continents.

 Australia has approximately 123 million hectares of native forest, which represents about 16% of Australia's land area.

 The majority of Australia's trees are hardwoods, typically eucalypts, rather than softwoods like pine.

 While softwoods dominate some native forests, their total area is judged insufficient to constitute a major forest type in Australia's National Forest
Inventory.

 The Forests Australia website provides up-to-date information on Australia's forests.

 Detailed information on Australia's forests is available from Australia's State of the Forests Reports that are published every five years.
</item 6>

<item 7>
Anigozanthos humilis
Anigozanthos humilis is a species of "Anigozanthos" in the family Haemodoraceae, known as common cat's paw or catspaw.

 This flowering perennial plant is endemic to Southwest Australia and widespread in its open forests.
</item 7>

<item 8>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 8>

<item 9>
Anigozanthos
Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.

 The 11 species and several subspecies are commonly known as kangaroo paw and catspaw depending on the shape of their flowers.

 A further species, previously identified as "Anigozanthos fuliginosus" and commonly known as the black kangaroo paw, has been transferred to its own monotypic
genus and is now known as "Macropidia fuliginosa".
</item 9>

<item 10>
Anigozanthos preissii
Anigozanthos preissii, the Albany cat's paw, is a herbaceous plant species in the family Haemodoraceae, endemic to Western Australia.
</item 10>
=== END ITEM id=b8e39684419a0363 ===

=== ITEM id=5003195407214e4f ===
Gold sentences and facts:
Sentence 0: Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.
  Fact 0.0: Claudio Giovanni Antonio Monteverdi was baptized on 15 May 1567.
  Fact 0.1: Claudio Giovanni Antonio Monteverdi died on 29 November 1643.
  Fact 0.2: Claudio Giovanni Antonio Monteverdi was an Italian composer.
  Fact 0.3: Claudio Giovanni Antonio Monteverdi was an Italian string player.
  Fact 0.4: Claudio Giovanni Antonio Monteverdi was an Italian choirmaster.
Sentence 1:  A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
  Fact 1.0: Monteverdi composed both secular and sacred music.
  Fact 1.1: Monteverdi was a pioneer in the development of opera.
  Fact 1.2: Monteverdi is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
Sentence 2: Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.
  Fact 2.0: Georges Bizet was born on 25 October 1838.
  Fact 2.1: Georges Bizet died on 3 June 1875.
  Fact 2.2: Georges Bizet was registered at birth as Alexandre César Léopold Bizet.
  Fact 2.3: Georges Bizet was a French composer.
  Fact 2.4: Georges Bizet was a composer of the romantic era.
Sentence 3:  Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the most popular and frequently performed works in the entire opera repertoire.
  Fact 3.0: Georges Bizet is best known for his operas.
  Fact 3.1: Bizet's career was cut short by his early death.
  Fact 3.2: Bizet achieved few successes before his final work.
  Fact 3.3: Bizet's final work is 'Carmen'.
  Fact 3.4: Carmen has become one of the most popular works in the entire opera repertoire.
  Fact 3.5: Carmen is one of the most frequently performed works in the entire opera repertoire.

Retrieved items:
<item 1>
Symphony in C (Bizet)
The Symphony in C is an early work by the French composer Georges Bizet.

 According to "Grove's Dictionary", the symphony "reveals an extraordinarily accomplished talent for a 17-year-old student, in melodic invention, thematic
handling and orchestration."

 Bizet started work on the symphony in C major on 29 October 1855, four days after turning 17, and finished it roughly a month later.

 It was written while he was studying at the Paris Conservatoire under the composer Charles Gounod, and was evidently a student assignment.

 Bizet showed no apparent interest in having it performed or published, and the piece was never played in his lifetime.

 He used certain material from the symphony in later works, however.

 There is no mention of the work in Bizet's letters, and it was unknown to his earlier biographers.

 His widow, Geneviève Halévy (1849–1926), gave the manuscript to Reynaldo Hahn, who left it along with other papers to the archives of the conservatory library,
where it was found in 1933 by Jean Chantavoine.
</item 1>

<item 2>
Roma Symphony (Bizet)
The Symphony in C "Roma" is the second of Georges Bizet's symphonies.

 Unlike his first symphony, also in C major, which was written quickly at the age of 17, "Roma" was written over an eleven-year span, between the ages of 22 and
33 (he died at age 36).

 Bizet was never fully satisfied with it, subjecting it to a number of revisions, but died before finishing his definitive version.

 All four movements were performed in his lifetime, but never all on the same occasion.

 The full symphony in its latest revision was premiered in 1875, after his death.

 It is perhaps because of Bizet's dissatisfaction that the work is often said to be "unfinished".

 However, in the form in which it exists today, it is a complete work and is fully scored.

 It has been recorded a number of times but is not often heard on the concert platform.
</item 2>

<item 3>
L'Orfeo discography
These lists show the audio and visual recordings of the opera "L'Orfeo" by Claudio Monteverdi.

 The opera was first performed in Mantua in 1607, at the court of Duke Vincenzo Gonzaga, and is one of the earliest of all operas.

 The first recording of "L'Orfeo" was issued in 1939, a freely adapted version of Monteverdi's music edited by Giacomo Benvenuti, given by the orchestra of La
Scala Milan conducted by Ferrucio Calusio.

 In 1949 the Berlin Radio Orchestra under Helmut Koch recorded the complete opera, on long-playing records (LPs).

 The advent of LP recordings was, as Harold Schonberg later wrote, an important factor in the postwar revival of interest in Renaissance and Baroque music, and
from the mid-1950s recordings of "L'Orfeo" have been issued on many labels.

 Koch's landmark version was reissued in 1962, when it was compared unfavourably with others that had by then been issued.
</item 3>

<item 4>
John Whenham
John Whenham is an English musicologist and academic who specializes in early Italian baroque music.

 He earned both a Bachelor of Music and a Master of Music from the University of Nottingham, and a Doctor of Philosophy from the University of Oxford.

 He is a leading expert on the life and works of Claudio Monteverdi, and is the author of the books "Duet and Dialogue in the Age of Monteverdi" (Ann Arbor,
Michigan: University Microfilms International, 1982) "Monteverdi, 'Orfeo' " (London: Cambridge University Press, 1986), "Monteverdi, Vespers (1610)" (Cambridge
University Press, 1997), and "The Cambridge Companion to Monteverdi" (with Richard Wistreich, Cambridge University Press, 2007).

 For five years he was co-editor of the journal "Music & Letters".

 He currently serves on the board of the Birmingham Early Music Festival and is head of the music history department at the University of Birmingham.
</item 4>

<item 5>
Stattkus-Verzeichnis
The Stattkus-Verzeichnis (SV) is a catalogue of the musical compositions of the Italian composer Claudio Monteverdi.

 The catalogue was published in 1985 by Manfred H. Stattkus ("Claudio Monteverdi: Verzeichnis der erhaltenen Werke").

 A free, basic second edition of the catalogue is available online.
</item 5>

<item 6>
L'Arlésienne (Bizet)
Georges Bizet composed L'Arlésienne as incidental music to Alphonse Daudet's play of the same name, usually translated as "The Girl from Arles".

 It was first performed on 1 October 1872 at the Vaudeville Theatre (now a cinema known as the Gaumont Opéra).

 Bizet's music consists of 27 numbers (some only a few bars) for voice, chorus, and small orchestra, ranging from short solos to longer entr'actes. Bizet
himself played the harmonium backstage at the premiere performance.
</item 6>

<item 7>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 7>

<item 8>
Carmen Jones
Carmen Jones is a 1943 Broadway musical with music by Georges Bizet (orchestrated for Broadway by Robert Russell Bennett) and lyrics and book by Oscar
Hammerstein II which was performed at The Broadway Theatre.

 Conceptually, it is Bizet's opera "Carmen" updated to a World War II-era African-American setting.

 (Bizet's opera was, in turn, based on the 1846 novella by Prosper Mérimée.)

 The Broadway musical was produced by Billy Rose, using an all-black cast, and directed by Hassard Short.

 Robert Shaw prepared the choral portions of the show.
</item 8>

<item 9>
Georges Bizet
Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.

 Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the
most popular and frequently performed works in the entire opera repertoire.
</item 9>

<item 10>
Kamarinskaya
Kamarinskaya (Russian: камаринская ) is a Russian traditional folk dance, which is mostly known today as the Russian composer Mikhail Glinka's composition of
the same name.

 Glinka's "Kamarinskaya", written in 1848, was the first orchestral work based entirely on Russian folk song and to use the compositional principles of that
genre to dictate the form of the music.

 It became a touchstone for the following generation of Russian composers ranging from the Western-oriented Pyotr Ilyich Tchaikovsky to the group of
nationalists known collectively as The Five and was also lauded abroad, most notably by French composer Hector Berlioz.
</item 10>
=== END ITEM id=5003195407214e4f ===

=== ITEM id=ec7bbd3b0c2d01d4 ===
Gold sentences and facts:
Sentence 0:  He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.
  Fact 0.0: The writer has written three novels.
  Fact 0.1: The writer has written three collections of short stories.
  Fact 0.2: The writer has written a collection of novellas.
  Fact 0.3: The writer has written seven volumes of essays.
  Fact 0.4: Three of the writer's essay volumes have won National Book Critics Circle Award prizes.
  Fact 0.5: One of the writer's essay volumes, 'A Temple of Texts' (2006), won the Truman Capote Award for Literary Criticism.
Sentence 1: Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for Literature.
  Fact 1.0: Naguib Mahfouz was born on December 11, 1911.
  Fact 1.1: Naguib Mahfouz died on August 30, 2006.
  Fact 1.2: Naguib Mahfouz's Arabic name is نجيب محفوظ, transliterated Nagīb Maḥfūẓ.
  Fact 1.3: Naguib Mahfouz was an Egyptian writer.
  Fact 1.4: Naguib Mahfouz won the 1988 Nobel Prize for Literature.

Retrieved items:
<item 1>
The Beginning and the End (novel)
The Beginning and the End (بداية ونهاية) is a novel by Naguib Mahfouz, who was awarded the Nobel Prize in literature in 1988.

 An Egyptian, Naguib has been credited with modernizing Arabic literature, with his prolific writing style and his themes on existentialism.

 The novel is marked by very bold characterization by Naguib Mahfouz, for the time period that the story is set in.

 The story moves at a prolific pace and tries to look at the world from each characters view point.
</item 1>

<item 2>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 2>

<item 3>
Cairo Trilogy
The Cairo Trilogy (Arabic: الثلاثية‎ ‎ ("The Trilogy") or ثلاثية القاهرة ("The Cairo Trilogy")) is a trilogy of novels written by the Egyptian novelist and
Nobel Prize winner Naguib Mahfouz, and one of the prime works of his literary career.
</item 3>

<item 4>
Naguib Mahfouz
Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for
Literature.

 He is regarded as one of the first contemporary writers of Arabic literature, along with Tawfiq el-Hakim, to explore themes of existentialism.

 He published 34 novels, over 350 short stories, dozens of movie scripts, and five plays over a 70-year career.

 Many of his works have been made into Egyptian and foreign films.
</item 4>

<item 5>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 5>

<item 6>
Naguib Mahfouz Medal for Literature
The Naguib Mahfouz Medal for Literature is a literary award for Arabic literature.

 It is given to the best contemporary novel written in Arabic, but not available in English translation.

 The winning book is then translated into English, and published by American University in Cairo Press.

 It was first awarded in 1996 and is presented annually on December 11, the birthday of Nobel laureate Naguib Mahfouz, by the President of the American
University in Cairo.
</item 6>

<item 7>
Thebes at War
Thebes at War is an early novel by the Egyptian writer Naguib Mahfouz.

 It was originally published in Arabic in 1944.

 An English translation by Humphrey Davies appeared in 2003.

 The novel is one of several that Mahfouz wrote at the beginning of his career, with Pharaonic Egypt as their setting.

 Others in this series of novels include "Khufu's Wisdom" (1939) and "Rhadopis of Nubia" (1943).

 All have been translated into English and appeared in one volume under the title "Three Novels of Ancient Egypt" (Everyman's Library, 2007).
</item 7>

<item 8>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 8>

<item 9>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 9>

<item 10>
Bidaya wa Nihaya
Bidaya wa Nihaya (Arabic: بداية و نهاية‎ ‎ , English: A Beginning and an End ) is a 1960 Egyptian film directed by Salah Abouseif and based on the novel by the
same name.

 It was the first film adapted from a novel written by Nobel Prize winner Naguib Mahfouz.
</item 10>
=== END ITEM id=ec7bbd3b0c2d01d4 ===

=== ITEM id=b18cb4a2f3546e68 ===
Gold sentences and facts:
Sentence 0:  He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his early work as Knox Overstreet in "Dead Poets Society".
  Fact 0.0: The actor referred to as 'He' is best known for the role of Dan Rydell on "Sports Night."
  Fact 0.1: The actor referred to as 'He' is best known for the role of Will Gardner on "The Good Wife."
  Fact 0.2: The role of Will Gardner on "The Good Wife" earned the actor two Primetime Emmy Award nominations.
  Fact 0.3: The actor referred to as 'He' is best known for his early work as Knox Overstreet in "Dead Poets Society."
Sentence 1: Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.
  Fact 1.0: Dead Poets Society is a 1989 American drama film.
  Fact 1.1: Dead Poets Society was directed by Peter Weir.
  Fact 1.2: Dead Poets Society was written by Tom Schulman.
  Fact 1.3: Dead Poets Society stars Robin Williams.
Sentence 2:  Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students through his teaching of poetry.
  Fact 2.0: Dead Poets Society is set in 1959.
  Fact 2.1: Dead Poets Society is set at the fictional elite conservative Vermont boarding school Welton Academy.
  Fact 2.2: Dead Poets Society tells the story of an English teacher.
  Fact 2.3: The English teacher in Dead Poets Society inspires his students through his teaching of poetry.

Retrieved items:
<item 1>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 1>

<item 2>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 2>

<item 3>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 3>

<item 4>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 4>

<item 5>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 5>

<item 6>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 6>

<item 7>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 7>

<item 8>
Dead Poets Society
Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.

 Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students
through his teaching of poetry.
</item 8>

<item 9>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 9>

<item 10>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 10>
=== END ITEM id=b18cb4a2f3546e68 ===

=== ITEM id=81c5217ff8588e06 ===
Gold sentences and facts:
Sentence 0: Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.
  Fact 0.0: Giovanni 'Tinto' Brass was born on 26 March 1933.
  Fact 0.1: Giovanni 'Tinto' Brass is an Italian filmmaker.
Sentence 1: Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.
  Fact 1.0: Baltasar Kormakur Samper was born on 27 February 1966.
  Fact 1.1: Baltasar Kormakur Samper is an Icelandic actor.
  Fact 1.2: Baltasar Kormakur Samper is an Icelandic theater director.
  Fact 1.3: Baltasar Kormakur Samper is an Icelandic film director.
  Fact 1.4: Baltasar Kormakur Samper is an Icelandic film producer.

Retrieved items:
<item 1>
Baltasar Breki Samper
Baltasar Breki Samper (born 22 July 1989), sometimes referred to as Baltasar Breki Baltasarsson, is an Icelandic actor.

 He is the son of Icelandic director Baltasar Kormákur.

 Before graduating from the Icelandic Academy of the Arts in 2015 Samper helped his father on several films as second assistant director.

 He is known for his role as Hjörtur in the Icelandic TV series "Trapped".
</item 1>

<item 2>
Trasgredire
Trasgredire (Transgressing or Cheeky) is a 2000 sex comedy directed by Tinto Brass, with Yuliya Mayarchuk in the lead role.

 Certain parallelisms are drawn between "Nerosubianco" (1969), another Tinto Brass film set in London.
</item 2>

<item 3>
Fermo posta Tinto Brass
Fermo posta Tinto Brass is a 1995 Italian comedy film-erotic film directed by Tinto Brass and set in vignettes.
</item 3>

<item 4>
Stjórnin
Stjórnin comprised two singers, Grétar Örvarsson, born July 11, 1959 and Sigríður Beinteinsdóttir, born July 24, 1962.

 Together they represented Iceland at the Eurovision Song Contest 1990 held in Zagreb.

 Their entry, "Eitt lag enn" was placed fourth out of 22 songs.

 Both singers were later members of another Eurovision group Heart 2 Heart who represented Iceland in 1992 with the song "Nei eða já".

 This entry was placed seventh out of 23 in Malmö.
</item 4>

<item 5>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 5>

<item 6>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 6>

<item 7>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 7>

<item 8>
Dropout (1970 film)
Dropout is a 1970 Italian romantic drama directed by Tinto Brass.

 It stars real-life couple, Franco Nero and Vanessa Redgrave.

 They also worked with Brass a year later on the drama, "La vacanza".

 "Dropout" was released in France on December 18, 1970, followed by a theatrical release in Italy on February 22, 1971.
</item 8>

<item 9>
Contraband (2012 film)
Contraband is a 2012 American crime thriller film directed by Baltasar Kormákur, starring Mark Wahlberg, Kate Beckinsale, Ben Foster, Caleb Landry Jones,
Giovanni Ribisi, Lukas Haas, Diego Luna and J. K. Simmons.

 The film is a remake of the 2008 Icelandic film "Reykjavík-Rotterdam" which Baltasar Kormákur starred in.

 It was released on January 13, 2012 in the United States by Universal Pictures.
</item 9>

<item 10>
Baltasar Kormákur
Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.

 He is best known for directing the films "101 Reykjavík", "Hafið", "A Little Trip to Heaven" (starring Julia Stiles and Forest Whitaker), a film based on the
book "Mýrin" "(Jar City)" by Arnaldur Indriðason, "Contraband", "2 Guns" (starring Mark Wahlberg and Denzel Washington) and "Everest".

 His father is the Spanish painter Baltasar Samper and actor Baltasar Breki Samper is his son.
</item 10>
=== END ITEM id=81c5217ff8588e06 ===

=== ITEM id=4969ac5e46f84c95 ===
Gold sentences and facts:
Sentence 0:  The name comes from the City of Starachowice, where the factory is located.
  Fact 0.0: The name comes from the City of Starachowice.
  Fact 0.1: The factory is located in the City of Starachowice.
Sentence 1:  It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).
  Fact 1.0: The item was sold in various countries for many years.
  Fact 1.1: The item was sold in countries beyond the Eastern Bloc.
  Fact 1.2: The item was used by the Yemen Army.
Sentence 2: The Eastern Bloc was the group of socialist states of Central and Eastern Europe, generally the Soviet Union and the countries of the Warsaw Pact.
  Fact 2.0: The Eastern Bloc was the group of socialist states of Central and Eastern Europe.
  Fact 2.1: The Eastern Bloc generally included the Soviet Union.
  Fact 2.2: The Eastern Bloc generally included the countries of the Warsaw Pact.

Retrieved items:
<item 1>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 1>

<item 2>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 2>

<item 3>
E-Z Polish Factory
The E-Z Polish Factory is a building designed by Frank Lloyd Wright and is located in Chicago, Illinois, United States.

 Built in 1905, the E-Z Polish Company made polish for shoes and stoves.

 It now serves as practice space for local bands and artists.
</item 3>

<item 4>
Graph query row:
manufacturer: FSC Star
relationships: ['LOCATED_IN']
place: Starachowice
founded_on: None
country: None
</item 4>

<item 5>
Leader Trucks
Leader found their niche in the heavy transport market and built as a rigid 4x4 and 6x6 configuration for tray, tipper and agitator applications, they offered
reliable machines in the industries of earth-moving and off-road construction.

 Leader was notable for being the first manufacturer to fit Caterpillar engines to trucks, to fit automatic transmissions to diesel trucks in assembly, and the
first truck manufacturer in Australia to offer disc brakes.

 Leader boasted its components to be 80 percent Australian built with only the engine, transmission and steering box imported.

 By 1980, Leader had sold its 1000th truck in Australia and had also sold trucks to New Zealand, South Africa, Hong Kong, China and Indonesia.
</item 5>

<item 6>
FSC Star
Fabryka Samochodów Ciężarowych "Star" was a Polish truck manufacturer.

 The name comes from the City of Starachowice, where the factory is located.

 Their first vehicle was the "Star 20" in 1948.

 The most popular product was the "Star 266".

 The 266 model offered very good quality and powerful engines for a low price.

 It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).

 For many years FSC Star was a state-owned company.

 Star is now owned by MAN AG who eliminated the brand in January 2009.
</item 6>

<item 7>
Shaanxi Automobile Group
Shaanxi Automobile Group Co., Ltd. is a Chinese bus and truck manufacturer headquartered in Xi'an, Shaanxi.

 It employs around 23,000 employees.

 It manufactures heavy duty truck, buses (chassis), medium-size heavy truck and heavy duty axles for trucks.

 It utilizes Magna Steyr and MAN SE technologies.

 Its bus chassis are sold under the Eurostar Bus brand.
</item 7>

<item 8>
Daimler AG
Daimler AG (] ) is a German multinational automotive corporation.

 Daimler AG is headquartered in Stuttgart, Baden-Württemberg, Germany.

 As of 2014, Daimler owns or has shares in a number of car, bus, truck and motorcycle brands including Mercedes-Benz, Mercedes-AMG, Smart Automobile, Detroit
Diesel, Freightliner, Western Star, Thomas Built Buses, Setra, BharatBenz, Mitsubishi Fuso, MV Agusta as well as shares in Denza, KAMAZ, Beijing Automotive
Group.

 The luxury Maybach brand was terminated at the end of 2012, but revived in April 2015 as "Mercedes-Maybach" versions of the Mercedes-Benz S-Class and G-Class.

 In 2016 Daimler sold 3.0 million vehicles.

 By unit sales, Daimler is the thirteenth-largest car manufacturer and is the largest truck manufacturer in the world.

 In addition to automobiles, Daimler manufactures buses and provides financial services through its Daimler Financial Services arm.

 The company is a component of the Euro Stoxx 50 stock market index.
</item 8>

<item 9>
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 9>

<item 10>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 10>
=== END ITEM id=4969ac5e46f84c95 ===

=== ITEM id=411254bc428dad68 ===
Gold sentences and facts:
Sentence 0: Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.
  Fact 0.0: Cantarella was a poison
  Fact 0.1: Cantarella was allegedly used by the Borgias
  Fact 0.2: The alleged use of Cantarella by the Borgias occurred during the papacy of Pope Alexander VI
Sentence 1:  The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to remove his name from the records.
  Fact 1.0: The use of Cantarella is not well documented in any of the papal records
  Fact 1.1: Cantarella was most likely conceived after 1503
  Fact 1.2: Cantarella was most likely conceived as part of Pope Julius II's effort to remove his name from the records
Sentence 2: Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
  Fact 2.0: Damnatio memoriae is a modern Latin phrase
  Fact 2.1: Damnatio memoriae literally means condemnation of memory
  Fact 2.2: Damnatio memoriae means that a person must not be remembered

Retrieved items:
<item 1>
Cantarella
Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.

 It was probably a variation of arsenic or cantharidin powder (made from blister beetles) .

 The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to
remove his name from the records.
</item 1>

<item 2>
Sicut Dudum
Sicut Dudum (English: "Just As Long Ago" ) is a papal bull promulgated by Pope Eugene IV in Florence on January 13, 1435, which forbade the enslavement of local
natives in the Canary Islands who had converted or were converting to Christianity.

 "Sicut Dudum" was meant to reinforce "Creator Omnium", issued the previous year, condemning Portuguese slave raids in the Canary Islands.

 Over forty years after "Creator Omnium" and "Sicut Dudum, Pope Sixtus IV found it necessary repeat the prohibition in his papal bull "Regimini Gregis" which
threatened the excommunication of all captains or pirates who enslaved Christians.
</item 2>

<item 3>
Felice della Rovere
Felice della Rovere (c. 1483 – September 27, 1536), also known as Madonna Felice, was the illegitimate daughter of Pope Julius II.

 One of the most powerful women of the Italian Renaissance, she was born in Rome around 1483 to Lucrezia Normanni and Cardinal Giuliano della Rovere, later Pope
Julius II.

 Felice was well educated, became accepted into close courtly circles of aristocratic families, and formed friendships with scholars and poets through her
education and genuine interest in humanism.

 Through the influence of her father, including an arranged marriage to Gian Giordano Orsini, she wielded extraordinary wealth and influence both within and
beyond the Roman Curia.

 In particular, she negotiated a peace between Julius II and the Queen of France, and held the position of Orsini Signora for over a decade following the death
of her husband in 1517.

 Felice further increased her power through a castle that she bought with money received from her father, the Castle at Palo, and through her involvement in the
grain trade.
</item 3>

<item 4>
Pope Julius II
Pope Julius II (Italian: "Papa Giulio II" ; Latin: "Iulius II" ) (5 December 1443 – 21 February 1513), born Giuliano della Rovere, and nicknamed "The Fearsome
Pope" and "The Warrior Pope", was Pope from 1 November 1503 to his death in 1513.

 His papacy was marked by an active foreign policy, ambitious building projects, and patronage of the arts—he commissioned the destruction and rebuilding of St.
Peter's Basilica, and Michelangelo's decoration of the ceiling of the Sistine Chapel.

 In addition to an active military policy, he personally led troops into battle on at least two occasions, the first to expel Giovanni Bentivoglio from Bologna
(17 August 1506–23 March 1507), and the second in an attempt to recover Ferrara for the Papal States (1 September 1510–29 June 1512).
</item 4>

<item 5>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 5>

<item 6>
Papal conclave, September 1503
The papal conclave of September 1503 elected Pope Pius III to succeed Pope Alexander VI.

 Due to the Italian Wars, the College of Cardinals was surrounded by three potentially hostile armies, loyal to Louis XII of France, Ferdinand II of Aragon, and
Cesare Borgia (the cardinal-nephew of Alexander VI).
</item 6>

<item 7>
Portrait of Pope Julius II
Portrait of Pope Julius II is an oil painting of 1511–12 by the Italian High Renaissance painter Raphael.

 The portrait of Pope Julius II was unusual for its time and would carry a long influence on papal portraiture.

 From early in its life, it was specially hung at the pillars of the church of Santa Maria del Popolo, on the main route from the north into Rome, on feast and
high holy days.

 Giorgio Vasari, writing long after Julius' death, said that "it was so lifelike and true it frightened everyone who saw it, as if it were the living man
himself".
</item 7>

<item 8>
Dictatus papae
Dictatus papae is a compilation of 27 statements of powers arrogated to the Pope that was included in Pope Gregory VII's register under the year 1075.

 Some historians argue that it was written (or dictated) by Gregory VII himself; others argue that it has been inserted in the register at a later date, and
that it had a different origin.

 In 1087 Cardinal Deusdedit published a collection of decretals, dedicated to Pope Victor III, that embodied the law of the Church – Canon law – which he had
compiled from many sources, both legitimate and false (see Pseudo-Isidore).

 The "Dictatus papae" agrees so clearly and closely with this collection that some have argued the "Dictatus" must have been based on it; and so must be of a
later date of compilation and insertion in the papal register than 1087.
</item 8>

<item 9>
The Agony and the Ecstasy (film)
The Agony and the Ecstasy is a 1965 American film directed by Carol Reed, starring Charlton Heston as Michelangelo and Rex Harrison as Pope Julius II.

 The film was partly based on Irving Stone's biographical novel of the same name.

 This film deals with the conflicts of Michelangelo and Pope Julius II during the painting of the Sistine Chapel's ceiling.

 It also features a soundtrack co-written by prolific composers Alex North and Jerry Goldsmith.
</item 9>

<item 10>
Dum Diversas
Dum Diversas (English: "Until different") is a papal bull issued on 18 June 1452 by Pope Nicholas V.

 It authorized Afonso V of Portugal to conquer Saracens and pagans and consign them to "perpetual servitude".

 Pope Calixtus III reiterated the bull in 1456 with "Inter Caetera" (not to be confused with Alexander VI's), renewed by Pope Sixtus IV in 1481 and Pope Leo X
in 1514 with "Precelse denotionis".

 The concept of the consignment of exclusive spheres of influence to certain nation states was extended to the Americas in 1493 by Pope Alexander VI with "Inter
caetera".
</item 10>
=== END ITEM id=411254bc428dad68 ===

=== ITEM id=f9a08cd0552866bc ===
Gold sentences and facts:
Sentence 0: The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.
  Fact 0.0: The Shame of a City is a 2006 feature-length documentary.
  Fact 0.1: The Shame of a City premiered at the Philadelphia Film Festival.
Sentence 1: Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and preserved in the WPA Slave Narrative Collection.
  Fact 1.0: Unchained Memories is a 2003 documentary film.
  Fact 1.1: Unchained Memories is about the stories of former slaves.
  Fact 1.2: The former slaves were interviewed during the 1930s.
  Fact 1.3: The interviews were part of the Federal Writers' Project.
  Fact 1.4: The stories are preserved in the WPA Slave Narrative Collection.

Retrieved items:
<item 1>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 1>

<item 2>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 2>

<item 3>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 3>

<item 4>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 4>

<item 5>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 5>

<item 6>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 6>

<item 7>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 7>

<item 8>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 8>

<item 9>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 9>

<item 10>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 10>
=== END ITEM id=f9a08cd0552866bc ===

=== ITEM id=ce5bdbcdfe846c57 ===
Gold sentences and facts:
Sentence 0: George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
  Fact 0.0: George Albert Smith Sr. was born on April 4, 1870.
  Fact 0.1: George Albert Smith Sr. died on April 4, 1951.
  Fact 0.2: George Albert Smith Sr. was an American religious leader.
  Fact 0.3: George Albert Smith Sr. served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
Sentence 1: Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.
  Fact 1.0: Wong Jing was born on 3 May 1955.
  Fact 1.1: Wong Jing is a Hong Kong film director.
  Fact 1.2: Wong Jing is a Hong Kong film producer.
  Fact 1.3: Wong Jing is a Hong Kong actor.
  Fact 1.4: Wong Jing is a Hong Kong presenter.
  Fact 1.5: Wong Jing is a Hong Kong screenwriter.

Retrieved items:
<item 1>
Ba-Jia-Jiang
Ba Jia Jiang (八家將) originated from the Chinese folk beliefs and myths, and usually generally refers to eight members of the godly realm.

 The general understanding of the origin of Ba Jia Jiang is that it is derived from the existence of eight generals who performed exorcism of evil spirits for
the Wufu Emperor (五福大帝).

 These eight generals became revered as the gods of the underworld, and are represented as the bodyguards or attendants for the temples of the nether Gods such
as the Dongyue Emperor (東獄大帝), Yama (King of Hell, 閻羅王) and Cheng Huang (the City Gods, 城隍).

 Gradually Ba Jia Jiang evolved to appear also as bodyguards to Wang Ye (Royal Lord, 王爺) and Matsu (媽祖), and at many other temples.

 Later on, believers at those temples dressed up as Ba Jia Jiang in order to defend the Gods.
</item 1>

<item 2>
George A. Smith
George Albert Smith (June 26, 1817 – September 1, 1875) (known throughout his life as George A. Smith) was an early leader in the Latter-day Saint movement.

 He served in the Quorum of the Twelve Apostles and as a member of the First Presidency of The Church of Jesus Christ of Latter-day Saints (LDS Church).
</item 2>

<item 3>
Ba-Jia-Jiang
These actions evolved into Taiwanese folk activities, which are part of the Wu Array (Military Array, 武陣) in Din Tao (Taiwanese troupes, 陣頭).

 Ba Jia Jiang is responsible for the capture of ghosts and evil spirits, bringing safety and good luck, and providing protection.

 They contain a strong religious nature, and Din Tao (Taiwanese troupes, 陣頭) often seem as mysterious, threatening and serious.

 Some Ba Jia Jiang members have relations with Taiwanese gangsters.
</item 3>

<item 4>
D. Arthur Haycock
David Arthur Haycock (September 4, 1916 – February 25, 1994) was a personal secretary to several twentieth-century presidents of The Church of Jesus Christ of
Latter-day Saints (LDS Church), including George Albert Smith, Joseph Fielding Smith, Harold B. Lee, Spencer W. Kimball and Ezra Taft Benson.
</item 4>

<item 5>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 5>

<item 6>
Justin Smith (milliner)
Justin Smith (born 1978) is a leading British milliner based in London.

 He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.

 Smith's hats have been exhibited around the world, and have been acquired by such museums as the Fashion Institute of Technology in New York and the Victoria &
Albert Museum in London for Hats: An Anthology curated by Stephen Jones.

 His work is in continuous demand by stylists and photographers for the quality style press.

 He has been visiting lecturer at London's Royal College of Art and has conducted seminars and teaching workshops for the British Council.
</item 6>

<item 7>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 7>

<item 8>
Graph query row:
GeorgeAlbertSmith: George Albert Smith Sr.
church: The Church of Jesus Christ of Latter-day Saints
GeorgeAlbertSmithChurchPosition: eighth president
WongJing: Wong Jing
WongJingChurch: None
WongJingChurchRole: None
</item 8>

<item 9>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 9>

<item 10>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 10>
=== END ITEM id=ce5bdbcdfe846c57 ===

=== ITEM id=dd051a91ad526930 ===
Gold sentences and facts:
Sentence 0: Eryngium is a genus of flowering plants in the family Apiaceae.
  Fact 0.0: Eryngium is a genus of flowering plants.
  Fact 0.1: Eryngium is in the family Apiaceae.
Sentence 1: Arnebia is a genus of flowering plants in the family Boraginaceae.
  Fact 1.0: Arnebia is a genus of flowering plants.
  Fact 1.1: Arnebia is in the family Boraginaceae.

Retrieved items:
<item 1>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 1>

<item 2>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 2>

<item 3>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 3>

<item 4>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 4>

<item 5>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 5>

<item 6>
Cyanobiont
Several cyanobionts involved with fungi and marine organisms also belong to the genera "Richelia, Calothrix, Synechocystis, Aphanocapsa" and "Anabaena", as well
as the species "Oscillatoria spongeliae".

 Although there are many documented symbioses between cyanobacteria and marine organisms, little is known about the nature of many of these symbioses.

 The possibility of discovering more novel symbiotic relationships is apparent from preliminary microscopic observations.
</item 6>

<item 7>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 7>

<item 8>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 8>

<item 9>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 9>

<item 10>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 10>
=== END ITEM id=dd051a91ad526930 ===

=== ITEM id=7e762adb03488bac ===
Gold sentences and facts:
Sentence 0: Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All Stars, as well of his own solo band, the Kuwata band.
  Fact 0.0: Keisuke Kuwata's name is written 桑田 佳祐 (Kuwata Keisuke).
  Fact 0.1: Keisuke Kuwata was born February 26, 1956.
  Fact 0.2: Keisuke Kuwata is a Japanese multi-instrumentalist.
  Fact 0.3: Keisuke Kuwata is a Japanese singer-songwriter.
  Fact 0.4: Keisuke Kuwata is frontman for the Southern All Stars.
  Fact 0.5: Keisuke Kuwata has his own solo band, the Kuwata band.
Sentence 1: Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.
  Fact 1.0: Moya Brennan was born Máire Ní Bhraonáin.
  Fact 1.1: Moya Brennan is also known as Máire Brennan.
  Fact 1.2: Moya Brennan was born 4 August 1952.
  Fact 1.3: Moya Brennan is an Irish folk singer.
  Fact 1.4: Moya Brennan is an Irish songwriter.
  Fact 1.5: Moya Brennan is an Irish harpist.
  Fact 1.6: Moya Brennan is an Irish philanthropist.

Retrieved items:
<item 1>
Bruce Yonemoto
Bruce Yonemoto (born 1949) is a Japanese-American multimedia artist.

 His work has been recognized by the National Endowment for the Arts, the American Film Institute, The Rockefeller Foundation, the Maya Deren Award for
Experimental Film and Video, and a mid-career survey show at the Japanese American National Museum, in addition to major solo exhibitions at the
InterCommunication Center in Tokyo, the Institute of Contemporary Art the University of Pennsylvania, and the Kemper Museum in Kansas City, MO.

 Yonemoto was featured in the 2002 Corcoran Gallery Bienniel, in Washington D.C. Yonemoto was also a recipient of the Creative Capital Visual Arts Award in
2008.
</item 1>

<item 2>
Kate Berlant
Kate Berlant (born July 16, 1987) is an American stand-up comedian, actress, and writer.

 She is known for appearing on "The Meltdown with Jonah and Kumail" and starring in an episode of the Netflix original series "".

 With John Early, Berlant created the Vimeo original series "555" produced by Abso Lutely Productions.

 Her father is the artist Tony Berlant.
</item 2>

<item 3>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 3>

<item 4>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 4>

<item 5>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 5>

<item 6>
Two Horizons
Two Horizons is the fifth solo album by Irish singer Moya Brennan.

 It is her first full-length release under the name "Moya Brennan", as opposed to "Máire Brennan".

 The album was predominantly recorded in her home studio in Dublin and was nominated for a Grammy award.

 The album was recorded between 2002–2003 and first became available on 23 October 2003.

 It is also Brennan's first non-Christian album since her 1993 recording Misty Eyed Adventures.

 It is the most successful of her solo albums to date and the second to be Grammy-nominated.
</item 6>

<item 7>
Óró – A Live Session
Óró – A Live Session is a music album by Irish musician Máire Brennan, now known as "Moya Brennan".

 Just before the start of the German tour, Moya and her band recorded a "live session" album to have available for fans on the tour.

 She's tried to recreate some of the live feel with the band in the studio covering many of the songs she plays (and was about to play) in concert.

 The album will become more widely available subsequently, but originally was released in 2005.
</item 7>

<item 8>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 8>

<item 9>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 9>

<item 10>
My Match Is a Makin'
My Match Is A Makin' (English translation of opening track title) is a music album by Irish musicians Moya Brennan and Cormac de Barra.

 This is Moya's eighth studio album to be released.

 It was released on 14 April 2010 exclusively to concert goers on her Spring 2010 tour of the Netherlands.

 The album was re-released to a wider audience under the title Voices & Harps.
</item 10>
=== END ITEM id=7e762adb03488bac ===

=== ITEM id=d1b978836a2987a8 ===
Gold sentences and facts:
Sentence 0: Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and medical teaching facilities in Saint John, New Brunswick.
  Fact 0.0: Dalhousie University, commonly known as Dal, is a public research university.
  Fact 0.1: Dalhousie University is in Nova Scotia, Canada.
  Fact 0.2: Dalhousie University has three campuses in Halifax.
  Fact 0.3: Dalhousie University has a fourth campus in Bible Hill.
  Fact 0.4: Dalhousie University has medical teaching facilities in Saint John, New Brunswick.
Sentence 1: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 1.0: California State Polytechnic University, Pomona is also known as CPP, Cal Poly Pomona, or Cal Poly.
  Fact 1.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 1.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.

Retrieved items:
<item 1>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 1>

<item 2>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 2>

<item 3>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 3>

<item 4>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 6>

<item 7>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 7>

<item 8>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 8>

<item 9>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 9>

<item 10>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 10>
=== END ITEM id=d1b978836a2987a8 ===
