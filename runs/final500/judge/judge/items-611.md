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

=== ITEM id=4f93de6646610d61 ===
Gold sentences and facts:
Sentence 0: Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in England to Australian parents.
  Fact 0.0: Ralph Foster Smart was born on 27 August 1908.
  Fact 0.1: Ralph Foster Smart died on 12 February 2001.
  Fact 0.2: Ralph Foster Smart was a film producer.
  Fact 0.3: Ralph Foster Smart was a television producer.
  Fact 0.4: Ralph Foster Smart was a director.
  Fact 0.5: Ralph Foster Smart was a writer.
  Fact 0.6: Ralph Foster Smart was born in England.
  Fact 0.7: Ralph Foster Smart's parents were Australian.
Sentence 1: Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.
  Fact 1.0: Randolph Severn 'Trey' Parker III was born on October 19, 1969.
  Fact 1.1: Randolph Severn 'Trey' Parker III is an American actor.
  Fact 1.2: Randolph Severn 'Trey' Parker III is an animator.
  Fact 1.3: Randolph Severn 'Trey' Parker III is a writer.
  Fact 1.4: Randolph Severn 'Trey' Parker III is a director.
  Fact 1.5: Randolph Severn 'Trey' Parker III is a producer.
  Fact 1.6: Randolph Severn 'Trey' Parker III is a singer.
  Fact 1.7: Randolph Severn 'Trey' Parker III is a songwriter.

Retrieved items:
<item 1>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 1>

<item 2>
Ralph Smart
Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in
England to Australian parents.
</item 2>

<item 3>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 3>

<item 4>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 4>

<item 5>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 5>

<item 6>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 6>

<item 7>
Trey Parker
Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.

 He is known for co-creating "South Park" (1997–present) along with his creative partner Matt Stone, as well as co-writing and co-directing the Tony
Award-winning musical "The Book of Mormon" (2011).

 Parker was interested in film and music as a child, and attended the University of Colorado, Boulder following high school, where he met Stone.

 The two collaborated on various short films, and starred in a feature-length musical, titled "Cannibal!

 The Musical" (1993).
</item 7>

<item 8>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 8>

<item 9>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 9>

<item 10>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 10>
=== END ITEM id=4f93de6646610d61 ===

=== ITEM id=f3bdd082bbb642f6 ===
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
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 1>

<item 2>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 2>

<item 3>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 3>

<item 4>
Pterostyrax hispidus
Pterostyrax hispidus, the epaulette tree or fragrant epaulette tree, is a species of flowering plant in the family Styracaceae, native to China and Japan.

 Growing to 15 m tall by 12 m broad, it is a substantial, spreading, deciduous shrub with oval leaves up to 20 cm long, and clusters of pure white, fragrant,
pendent flowers in summer.
</item 4>

<item 5>
Rhabdodontidae
Rhabdodontids were herbivorous ornithopod dinosaurs from the Cretaceous Period.

 Rhabdodontids were similar to large, robust "hypsilophodonts", with deep skulls and jaws.

 The family was first proposed by David B. Weishampel and colleagues in 2002.

 The rhabdodontids were defined as "the most recent common ancestor of "Zalmoxes robustus" and "Rhabdodon priscus" and all the descendants of this common
ancestor".

 In 2005, Paul Sereno defined the family as "the most inclusive clade containing "Rhabdodon priscus" but not "Parasaurolophus walkeri"".

 The Rhabdodontidae includes the type genus "Rhabdodon", "Zalmoxes", "Mochlodon" and possibly "Muttaburrasaurus".

 Rhabdodontid fossils have been found in Europe and Australia in formations dating from the Early to Late Cretaceous, 100 to 66 million years ago.

 Ősi "et al."
</item 5>

<item 6>
Dregea
Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.

 It is native to Africa and southern Asia.
</item 6>

<item 7>
Dregea sinensis
Dregea sinensis is a species of plant in the Apocynaceae family that is native to China.
</item 7>

<item 8>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 8>

<item 9>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 9>

<item 10>
Styrax fraserensis
Styrax fraserensis is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is a tree endemic to Peninsular Malaysia.

 It is threatened by habitat loss.
</item 10>
=== END ITEM id=f3bdd082bbb642f6 ===

=== ITEM id=823aab9a6301d9f1 ===
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
John Whenham
John Whenham is an English musicologist and academic who specializes in early Italian baroque music.

 He earned both a Bachelor of Music and a Master of Music from the University of Nottingham, and a Doctor of Philosophy from the University of Oxford.

 He is a leading expert on the life and works of Claudio Monteverdi, and is the author of the books "Duet and Dialogue in the Age of Monteverdi" (Ann Arbor,
Michigan: University Microfilms International, 1982) "Monteverdi, 'Orfeo' " (London: Cambridge University Press, 1986), "Monteverdi, Vespers (1610)" (Cambridge
University Press, 1997), and "The Cambridge Companion to Monteverdi" (with Richard Wistreich, Cambridge University Press, 2007).

 For five years he was co-editor of the journal "Music & Letters".

 He currently serves on the board of the Birmingham Early Music Festival and is head of the music history department at the University of Birmingham.
</item 1>

<item 2>
Georges Bizet
Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.

 Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the
most popular and frequently performed works in the entire opera repertoire.
</item 2>

<item 3>
L'Arlésienne (Bizet)
Georges Bizet composed L'Arlésienne as incidental music to Alphonse Daudet's play of the same name, usually translated as "The Girl from Arles".

 It was first performed on 1 October 1872 at the Vaudeville Theatre (now a cinema known as the Gaumont Opéra).

 Bizet's music consists of 27 numbers (some only a few bars) for voice, chorus, and small orchestra, ranging from short solos to longer entr'actes. Bizet
himself played the harmonium backstage at the premiere performance.
</item 3>

<item 4>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 4>

<item 5>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 5>

<item 6>
L'Orfeo discography
These lists show the audio and visual recordings of the opera "L'Orfeo" by Claudio Monteverdi.

 The opera was first performed in Mantua in 1607, at the court of Duke Vincenzo Gonzaga, and is one of the earliest of all operas.

 The first recording of "L'Orfeo" was issued in 1939, a freely adapted version of Monteverdi's music edited by Giacomo Benvenuti, given by the orchestra of La
Scala Milan conducted by Ferrucio Calusio.

 In 1949 the Berlin Radio Orchestra under Helmut Koch recorded the complete opera, on long-playing records (LPs).

 The advent of LP recordings was, as Harold Schonberg later wrote, an important factor in the postwar revival of interest in Renaissance and Baroque music, and
from the mid-1950s recordings of "L'Orfeo" have been issued on many labels.

 Koch's landmark version was reissued in 1962, when it was compared unfavourably with others that had by then been issued.
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
Symphony in C (Bizet)
Soon thereafter, Bizet's first British biographer Douglas Charles Parker (1885–1970) showed the manuscript to the conductor Felix Weingartner, who led the first
performance in Basel, Switzerland, on 26 February 1935.
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
=== END ITEM id=823aab9a6301d9f1 ===

=== ITEM id=cb28617b5c9a691f ===
Gold sentences and facts:
Sentence 0: John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).
  Fact 0.0: John H. Garabedian is an American radio personality.
  Fact 0.1: John H. Garabedian is an American disc jockey.
  Fact 0.2: John H. Garabedian was born on December 20, 1941.
  Fact 0.3: John H. Garabedian is age 75.
Sentence 1: Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
  Fact 1.0: Charles A. Agemian was born July 22, 1909.
  Fact 1.1: Charles A. Agemian's life ended March 30, 1996.
  Fact 1.2: Charles A. Agemian was an American banker.
  Fact 1.3: Charles A. Agemian took early retirement from Chase Manhattan Bank.
  Fact 1.4: Charles A. Agemian was executive vice president of operations at Chase Manhattan Bank.
  Fact 1.5: Charles A. Agemian became chairman of the Hackensack Trust Co.
  Fact 1.6: Charles A. Agemian became chief executive officer of the Hackensack Trust Co.
  Fact 1.7: The Hackensack Trust Co. was later renamed Garden State National Bank.

Retrieved items:
<item 1>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 1>

<item 2>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 2>

<item 3>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 3>

<item 4>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 4>

<item 5>
Jimmy O'Neill (DJ)
James Franklin O'Neill (January 8, 1940 – January 11, 2013) was an American DJ and broadcaster who hosted the ABC television show "Shindig!

" from 1964-1966.

 O'Neill was owner of Pandora's Box, an influential Sunset Strip music venue in West Hollywood, California that was the center of the 1966 Sunset Strip curfew
riots.
</item 5>

<item 6>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 6>

<item 7>
Radio personality
A radio personality (American English) or radio presenter (British English), commonly referred to as a "disc jockey" or "DJ" for short, is a person who has an
on-air position in radio broadcasting.

 A radio personality that hosts a radio show is also known as a radio host, and in India and Pakistan as a radio jockey.

 Radio personalities who introduce and play individual selections of recorded music are known as disc jockeys.

 The term has evolved to also describe a person who mixes a continuous flow of recorded music in real time.

 Broadcast radio personalities may include talk radio hosts, AM/FM radio show hosts, and satellite radio program hosts.

 Notable radio personalities include pop music radio hosts Martin Block, Alan Freed, Dick Clark, Delilah Luke, Ameen Sayani, Wolfman Jack, and Casey Kasem,
shock jocks such as Don Imus and Howard Stern, as well as sports talk hosts such as Mike Francesa and political talk hosts such as Rush Limbaugh.
</item 7>

<item 8>
Don Sherwood (DJ)
Don Sherwood (September 7, 1925 – November 6, 1983) was an American radio personality.

 He was a San Francisco, California, disc jockey during the 1950s and 1960s.

 Billed as "The World's Greatest Disc Jockey," Sherwood spent most of his career hosting a 6-9 a.m. weekday program on KSFO in San Francisco (560 kHz, 5000
watts), which was then owned by the singing cowboy actor Gene Autry.
</item 8>

<item 9>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 9>

<item 10>
Big Wilson
Big Wilson (born Malcolm John Wilson, Jr.; October 3, 1924 – October 5, 1989) was an American radio personality.

 He worked as a disc jockey at WNBC AM in New York City from the early 1960s until 1974 and moved to Miami in 1975 where he worked for WIOD and WCIX-TV.

 He was one of the last two hosts of the NBC network program Monitor.

 Prior to moving to NYC, Wilson and his wife Jody lived in a riverfront home in Rocky River, Ohio.

 He commuted to Cleveland where he was the leading radio personality for some years.

 Tim Conway and other well-known persons visited their home.
</item 10>
=== END ITEM id=cb28617b5c9a691f ===

=== ITEM id=924aebd88121665e ===
Gold sentences and facts:
Sentence 0: Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.
  Fact 0.0: The Dogo Cubano is also known as the Cuban Mastiff.
  Fact 0.1: The Dogo Cubano is also known as the Cuban Dogo.
  Fact 0.2: The Dogo Cubano is also known as the Cuban Dogge.
  Fact 0.3: The Dogo Cubano is an extinct dog breed.
  Fact 0.4: The Dogo Cubano is a dog breed from Cuba.
Sentence 1: The Dutch Shepherd is a herding dog of Dutch origin.
  Fact 1.0: The Dutch Shepherd is a herding dog.
  Fact 1.1: The Dutch Shepherd is a herding dog of Dutch origin.

Retrieved items:
<item 1>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 1>

<item 2>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 2>

<item 3>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 3>

<item 4>
Cuban Criollo
The Criollo Cubano originates from Spanish horses brought to Cuba by Diego Velázquez in 1751.

 There are four different breeds known as criollo: the Cubano de Paso, the Pinto Cubano, the Criollo de Trote and the Patibarcino.

 It is a small stout breed used by the Guajiro people.
</item 4>

<item 5>
Dogo Cubano
Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.

 It was of the Bull Mastiff type and was used for dog fighting.
</item 5>

<item 6>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 6>

<item 7>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 7>

<item 8>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 8>

<item 9>
Graph query row:
entity: Dutch Shepherd
kind: breed
relationship: LOCATED_IN
related: Belgium
related_kind: country
</item 9>

<item 10>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 10>
=== END ITEM id=924aebd88121665e ===

=== ITEM id=0ee798155604a4a0 ===
Gold sentences and facts:
Sentence 0:  The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
  Fact 0.0: The show originally ran from 7 September 1977 to 20 November 1995.
  Fact 0.1: The show was hosted by Gordon Burns.
  Fact 0.2: The show was usually broadcast on the ITV network.
  Fact 0.3: The show was usually broadcast on Mondays at 7pm.
Sentence 1:  He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West Tonight" from January 1997 to October 2011.
  Fact 1.0: The person referred to as 'He' was the host of 'The Krypton Factor' for its original 18-year run.
  Fact 1.1: The Krypton Factor's original run was 1977–1995.
  Fact 1.2: The person referred to as 'He' was the chief anchorman for the BBC regional news programme 'North West Tonight'.
  Fact 1.3: The person referred to as 'He' was the chief anchorman for 'North West Tonight' from January 1997 to October 2011.

Retrieved items:
<item 1>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 1>

<item 2>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 2>

<item 3>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 3>

<item 4>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 4>

<item 5>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 5>

<item 6>
Gordon Burns (footballer)
Gordon Burns (born 2 December 1978) is a Scottish former professional footballer who is currently manager of Troon in the Scottish Junior Football Association,
West Region.

 He has previously played in the Scottish Championship for Ayr United.
</item 6>

<item 7>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 7>

<item 8>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 8>

<item 9>
Password (UK game show)
Password was a British panel game show based on the US version of the same name.

 It was originally aired on ITV produced by ATV from 12 March to 10 September 1963 hosted by Shaw Taylor, then it aired on BBC2 from 24 March to 28 April 1973
hosted by Brian Redhead before moving to its flagship channel BBC1 from 7 January 1974 to 1976 first hosted by Eleanor Summerfield then by Esther Rantzen, it
was then aired on Channel 4 produced by Thames in association with Talbot Television and Goodson-Todman Productions from 6 November 1982 to 14 May 1983 hosted
by Tom O'Connor and then finally aired back on ITV produced by Ulster from 22 July 1987 to 5 August 1988 hosted by Gordon Burns.
</item 9>

<item 10>
The Krypton Factor
The Krypton Factor is a British game show produced by Granada for broadcast on ITV.

 The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
</item 10>
=== END ITEM id=0ee798155604a4a0 ===

=== ITEM id=3e81842440eb2208 ===
Gold sentences and facts:
Sentence 0: Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.
  Fact 0.0: Willi Forst was born Wilhelm Anton Frohs.
  Fact 0.1: Willi Forst's dates are 7 April 1903 to 11 August 1980.
  Fact 0.2: Willi Forst was an Austrian actor.
  Fact 0.3: Willi Forst was a screenwriter.
  Fact 0.4: Willi Forst was a film director.
  Fact 0.5: Willi Forst was a film producer.
  Fact 0.6: Willi Forst was a singer.
Sentence 1: Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.
  Fact 1.0: Elmer Clifton's dates are March 14, 1890 to October 15, 1949.
  Fact 1.1: Elmer Clifton was an American writer.
  Fact 1.2: Elmer Clifton was an American director.
  Fact 1.3: Elmer Clifton was an American actor.
  Fact 1.4: Elmer Clifton was active from the early silent days.

Retrieved items:
<item 1>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 1>

<item 2>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 2>

<item 3>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 3>

<item 4>
Kaiserjäger (film)
Kaiserjäger is a 1956 Austrian film directed by Willi Forst.
</item 4>

<item 5>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 5>

<item 6>
Winfred Overholser
Winfred Overholser (1892 – October 6, 1964) was an American psychiatrist, president of the American Psychiatric Association, and for 25 years the superintendent
of St. Elizabeths Hospital, a federal institution for the mentally ill in Washington, D.C.
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 8>

<item 9>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 9>

<item 10>
Al Ulmer
Alfred C. Ulmer, Jr. was born in Jacksonville, Florida in August 1916.

 He was of Swiss extraction on his father's side, his father having been born in Zurich.

 Ulmer graduated from Princeton University in 1939 and joined the United States Navy prior to the start of World War II, ultimately becoming a major head of
intelligence operations during World War II.

 He married Doris Gibson Bridges and had three sons (Alfred III, James and Nicolas) and a daughter (Marguerite).

 He received the Intelligence Medal of Merit when he retired from his position in 1962.

 Ulmer then went on to business and in the 1980s joined the Swiss banking firm Lombard Odier et Cie. in Geneva, later setting up Lombard Odier's operations in
Bermuda.

 He died on June 22, 2000 in Virginia Beach, Virginia.
</item 10>
=== END ITEM id=3e81842440eb2208 ===

=== ITEM id=7134ad9f2fdebccf ===
Gold sentences and facts:
Sentence 0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and February 1955.
  Fact 0.0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records.
  Fact 0.1: Dean Martin recorded Swingin' Down Yonder during three sessions.
  Fact 0.2: The sessions for Swingin' Down Yonder took place in September 1954, October 1954 and February 1955.
Sentence 1:  One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless charisma and self-assurance.
  Fact 1.0: Dean Martin was one of the most popular and enduring American entertainers of the mid-20th century.
  Fact 1.1: Dean Martin was nicknamed the 'King of Cool'.
  Fact 1.2: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless charisma.
  Fact 1.3: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless self-assurance.

Retrieved items:
<item 1>
Swingin' Down Yonder
Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and
February 1955.

 According to the original sleeve notes, all the songs have a "common geographical root: the American South."

 In 1963, Capitol Records re-released "Swingin' Down Yonder" under the titled "Southern Style".
</item 1>

<item 2>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 2>

<item 3>
Things Ain't What They Used to Be
In 1941 there was a strike against the American Society of Composers, Authors and Publishers, of which Duke Ellington was a member.

 Because of the strike he could not air his songs on the radio.

 Instead, he used songs written by his son Mercer and pianist Billy Strayhorn.

 Strayhorn's compositions of this time include "Take the 'A' Train", "Chelsea Bridge" and "Day Dream", and Mercer wrote "Things Ain't What They Used to Be",
"Blue Serge" and "Moon Mist".
</item 3>

<item 4>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 4>

<item 5>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 5>

<item 6>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 6>

<item 7>
Graph query row:
person: Dean Martin
nicknames: ['Dino Paul Crocetti', 'King of Cool']
</item 7>

<item 8>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 8>

<item 9>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 9>

<item 10>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 10>
=== END ITEM id=7134ad9f2fdebccf ===

=== ITEM id=f8db90fe163a6f92 ===
Gold sentences and facts:
Sentence 0:  is a board game created by Glenn Drover and published by Eagle Games in 2003.
  Fact 0.0: The board game in this sentence was created by Glenn Drover.
  Fact 0.1: The board game in this sentence was published by Eagle Games in 2003.
Sentence 1: Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.
  Fact 1.0: Can't Stop is a board game designed by Sid Sackson.
  Fact 1.1: Can't Stop was originally published by Parker Brothers in 1980.
  Fact 1.2: Can't Stop was long out of print in the United States.

Retrieved items:
<item 1>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 1>

<item 2>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 2>

<item 3>
Atari Corporation
Atari Corporation was an American manufacturer of computers and video game consoles from 1984 to 1996.

 Atari Corp. was founded in July 1984 when Warner Communications sold the home computing and game console divisions of Atari, Inc. to Jack Tramiel.

 Its chief products were the Atari ST, , Atari 7800, Atari Lynx, and Atari Jaguar.

 The company reverse merged with JTS Inc. in 1996, becoming a small division, which itself closed when JTS liquidated the IP to Hasbro Interactive in 1998.
</item 3>

<item 4>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 4>

<item 5>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 5>

<item 6>
Atari ST
The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.

 The initial ST model, the 520ST, saw limited release in April-June 1985 and was widely available in July.

 The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.

 The 1040ST, released in 1986, is the first personal computer to ship with a megabyte of RAM in the base configuration and also the first with a
cost-per-kilobyte of less than US$1.
</item 6>

<item 7>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 7>

<item 8>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 8>

<item 9>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 9>

<item 10>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 10>
=== END ITEM id=f8db90fe163a6f92 ===

=== ITEM id=48ccf19fa1ecef41 ===
Gold sentences and facts:
Sentence 0: Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in England to Australian parents.
  Fact 0.0: Ralph Foster Smart was born on 27 August 1908.
  Fact 0.1: Ralph Foster Smart died on 12 February 2001.
  Fact 0.2: Ralph Foster Smart was a film producer.
  Fact 0.3: Ralph Foster Smart was a television producer.
  Fact 0.4: Ralph Foster Smart was a director.
  Fact 0.5: Ralph Foster Smart was a writer.
  Fact 0.6: Ralph Foster Smart was born in England.
  Fact 0.7: Ralph Foster Smart's parents were Australian.
Sentence 1: Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.
  Fact 1.0: Randolph Severn 'Trey' Parker III was born on October 19, 1969.
  Fact 1.1: Randolph Severn 'Trey' Parker III is an American actor.
  Fact 1.2: Randolph Severn 'Trey' Parker III is an animator.
  Fact 1.3: Randolph Severn 'Trey' Parker III is a writer.
  Fact 1.4: Randolph Severn 'Trey' Parker III is a director.
  Fact 1.5: Randolph Severn 'Trey' Parker III is a producer.
  Fact 1.6: Randolph Severn 'Trey' Parker III is a singer.
  Fact 1.7: Randolph Severn 'Trey' Parker III is a songwriter.

Retrieved items:
<item 1>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 1>

<item 2>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 2>

<item 3>
Trey Parker
Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.

 He is known for co-creating "South Park" (1997–present) along with his creative partner Matt Stone, as well as co-writing and co-directing the Tony
Award-winning musical "The Book of Mormon" (2011).

 Parker was interested in film and music as a child, and attended the University of Colorado, Boulder following high school, where he met Stone.

 The two collaborated on various short films, and starred in a feature-length musical, titled "Cannibal!

 The Musical" (1993).
</item 3>

<item 4>
Rosie O'Donnell
Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.

 She has been a magazine editor and continues to be a celebrity blogger, a lesbian rights activist, a television producer, and a collaborative partner in the
LGBT family vacation company, R Family Vacations.
</item 4>

<item 5>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 5>

<item 6>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 6>

<item 7>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 7>

<item 8>
Ralph Smart
Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in
England to Australian parents.
</item 8>

<item 9>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 9>

<item 10>
Jerry Beck
Peabody and Sherman", DreamWorks' "Madagascar", and "Bee Movie".

 Beck is also an entertainment industry consultant for TV and home entertainment productions and releases related to classic cartoons and operates the blog
"Cartoon Research."

 He appears frequently as a documentary subject and audio commentator on releases of A&E's "Cartoons Go to War" as well as DVD collections of "Looney Tunes",
"Popeye the Sailor", and "Woody Woodpecker" cartoons, on which he serves as consultant and curator.
</item 10>
=== END ITEM id=48ccf19fa1ecef41 ===

=== ITEM id=a9824f8124cc9b4a ===
Gold sentences and facts:
Sentence 0: Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on the German island of Greifswalder Oie in December 1937.
  Fact 0.0: Operation Lighthouse was the name given to a failed experimental launch.
  Fact 0.1: The Operation Lighthouse launch involved four Aggregate 3 liquid-fuel rockets.
  Fact 0.2: Wernher von Braun conducted the Operation Lighthouse launch.
  Fact 0.3: Walter Dornberger conducted the Operation Lighthouse launch.
  Fact 0.4: The Operation Lighthouse launch took place on the German island of Greifswalder Oie.
  Fact 0.5: The Operation Lighthouse launch took place in December 1937.
Sentence 1: Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War II.
  Fact 1.0: Major-General Dr. Walter Robert Dornberger was born on 6 September 1895.
  Fact 1.1: Walter Robert Dornberger died on 27 June 1980.
  Fact 1.2: Walter Robert Dornberger was a German Army artillery officer.
  Fact 1.3: Walter Robert Dornberger's career spanned World War I.
  Fact 1.4: Walter Robert Dornberger's career spanned World War II.

Retrieved items:
<item 1>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 1>

<item 2>
Man Will Conquer Space Soon!
Man Will Conquer Space Soon!

 was the title of a famous series of 1950s magazine articles in "Collier's" detailing Wernher von Braun's plans for manned spaceflight.

 Edited by Cornelius Ryan, the individual articles were authored by such space notables of the time as Willy Ley, Fred Lawrence Whipple, Dr. Joseph Kaplan, Dr.
Heinz Haber, and von Braun.

 The articles were illustrated with paintings and drawings by Chesley Bonestell, Fred Freeman, and Rolf Klep, some of the finest magazine illustrators of the
time.
</item 2>

<item 3>
Von Braun Ferry Rocket
Von Braun Ferry Rocket was a concept design for a shuttle spacecraft that was developed by Wernher von Braun in a seminal series of early-1950s Collier's
magazine articles, Man Will Conquer Space Soon!

 by Wernher von Braun "et al." The Ferry Rocket concept has evolved over time.
</item 3>

<item 4>
The entries cover German rocket scientist Magnus “Mac” von Braun, Wernher’s brother; Wernher von Braun’s Ferry Rocket shuttle concept and the 1937 failed
Aggregate 3 launches (Operation Lighthouse); and Collier’s influential 1950s spaceflight series *Man Will Conquer Space Soon!*. They also profile German
astronauts Hans Schlegel, Ernst Messerschmid, Ulrich Walter, Sigmund Jähn, Ulf Merbold, and Soyuz
</item 4>

<item 5>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 5>

<item 6>
Magnus von Braun
Magnus "Mac" Freiherr von Braun (10 May 1919 – 21 June 2003) was a German chemical engineer, Luftwaffe aviator, and rocket scientist at Peenemünde, the
Mittelwerk, and after emigrating to the United States via Operation Paperclip, at Fort Bliss.

 He was the brother of Sigismund and Wernher von Braun.
</item 6>

<item 7>
The entries cover: lunar crater von Braun near Oceanus Procellarum, foreshortened from Earth; Antarctic Mount von Braun, named for Wernher von Braun;
Huntsville’s NASA- and defense-affiliated Von Braun Center; Saturn I’s Project Highwater, which studied high-altitude water release; Blue Origin’s stepwise VTVL
spaceflight program; black-hole superradiant instability (“bombs”), V404 Cygni and M33 X-7 binaries, erg
</item 7>

<item 8>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 8>

<item 9>
Von Braun (crater)
von Braun, named after the rocket pioneer Wernher von Braun, is a lunar crater located near the northwestern limb of the Moon.

 It lies along the western edge of the Oceanus Procellarum, to the northeast of the crater Lavoisier.

 The northeastern rim of this crater is on the edge of the Sinus Roris, a bay feature in the northwestern part of the Oceanus Procellarum.

 Due to its proximity to the limb, this crater appears significantly foreshortened when viewed from the Earth.
</item 9>

<item 10>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 10>
=== END ITEM id=a9824f8124cc9b4a ===

=== ITEM id=16802afc81b8d087 ===
Gold sentences and facts:
Sentence 0:  In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site LiveJasmin.
  Fact 0.0: In the category of pornographic websites, this site was surpassed in the rankings by xHamster.
  Fact 0.1: In the category of pornographic websites, this site was surpassed in the rankings by XVideos.
  Fact 0.2: In the category of pornographic websites, this site was surpassed in the rankings by Pornhub.
  Fact 0.3: In the category of pornographic websites, this site was surpassed in the rankings by the adult webcam site LiveJasmin.
  Fact 0.4: xHamster, XVideos, and Pornhub are competitor sites.
Sentence 1: XVideos is the world's largest pornographic video sharing and viewing website.
  Fact 1.0: XVideos is the world's largest pornographic video sharing and viewing website.

Retrieved items:
<item 1>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 1>

<item 2>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 2>

<item 3>
XHamster
xHamster is a pornographic media and social networking site headquartered in Limassol, Cyprus.

 xHamster serves user-submitted pornographic videos, webcam models, pornographic photographs, and erotic literature, and incorporates social networking
features.

 xHamster was founded in 2007.

 With more than 10 million members, it is the third most popular pornography website on the Internet after XVideos and Pornhub.
</item 3>

<item 4>
The entries survey magazines, broadcasters, websites, and media demographics. Publications range from children’s nature titles and women’s magazines to
political, religious, literary, and LGBTQ+ outlets; profiles include Rosie O’Donnell. Radio and TV examples span US local stations, regional Spectrum News
channels, and international youth/children’s networks. Adult-video sites include xHamster, YouPorn, and XVideos. Advertisers’ “key demographic” often means
adults 18–49, though targets vary
</item 4>

<item 5>
Pravda.ru
Pravda.ru (formerly Pravda Online) is a Russian internet news website established in 1999 and owned by Pravda.ru Holding (headed by Vadim Gorshenin).
</item 5>

<item 6>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 6>

<item 7>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 7>

<item 8>
YouPorn
YouPorn is a free pornographic video sharing website and one of the 100 most accessed websites in the world.

 Since launching in August 2006, it grew to become the most popular pornographic website on the internet, and, in November 2007, it was reported to be the
largest free pornographic website as well.

 As of February 2013, it was the 83rd most popular website overall and the fifth most popular pornographic website.

 In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site
LiveJasmin.
</item 8>

<item 9>
Viceland Canada, owned by Rogers, offers youth-oriented lifestyle documentaries and reality shows. Cartoon Network MENA and Africa serve children across their
regions; Africa was formerly HQ, separated from the UK feed in 1999 and independent by 2001. Adult sites: xHamster (founded 2007, Cyprus-based), YouPorn
(launched 2006), and XVideos (Polish-owned) share pornographic videos; xHamster and XVideos outranked YouPorn.
</item 9>

<item 10>
XY TV
It launched in December 2003 and distribution was limited to select cable companies.

 As of May 31, 2006, XY.tv has ceased operations, and is no longer broadcasting.

 It was wholly owned by American radio personality John Garabedian.
</item 10>
=== END ITEM id=16802afc81b8d087 ===
