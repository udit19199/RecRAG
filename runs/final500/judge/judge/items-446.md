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

=== ITEM id=5d1ef74e4a9f3e4c ===
Gold sentences and facts:
Sentence 0: of Montreal is an American experimental pop band from Athens, Georgia.
  Fact 0.0: of Montreal is an American band.
  Fact 0.1: of Montreal is an experimental pop band.
  Fact 0.2: of Montreal is from Athens, Georgia.
Sentence 1: Vib Gyor was a band that was formed in Leeds, England.
  Fact 1.0: Vib Gyor was a band.
  Fact 1.1: Vib Gyor was formed in Leeds, England.

Retrieved items:
<item 1>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 1>

<item 2>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 2>

<item 3>
The Gorgeous
The Gorgeous were a Canadian metalcore band formed in the summer of 2003 in Montreal.

 The band formed with Dali Shaw on guitar, Julien Brousseau on Bass and Miguel Shaw on drums.

 The three played a few shows without a vocalist and quickly recruited Jordon Daniel.
</item 3>

<item 4>
Fallen (Vib Gyor song)
"Fallen" was the first single from the Leeds band Vib Gyor.

 It was first given limited release in July 2006, and has attained critical acclaim and a large amount of radio play on both sides of the atlantic.
</item 4>

<item 5>
The Famines (band)
The Famines are a Canadian indie rock band formed in 2008 in Edmonton, Alberta, Canada now based in Montreal, Quebec, Canada.

 The two piece band uses a modern and minimalistic approach that draws comparison to mid 1970's protopunk and fuzzy garage rock.

The band name is meant to be a commentary on the continued feeling of emptiness and lacking in a society that is materially fulfilled.

 The band has two members, R. E. Biesinger on guitar and vocals, and Drew Demers on the drums.
</item 5>

<item 6>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 6>

<item 7>
Bloc Party
Bloc Party are an English indie rock band, currently composed of Kele Okereke (lead vocals, rhythm guitar, keyboards, sampler), Russell Lissack (lead guitar,
keyboards), Justin Harris (bass guitar, keyboards, saxophones, backing vocals) and Louise Bartle (drums, percussion).

 Former members Matt Tong and Gordon Moakes left the band in 2013 and 2015 respectively.

 Their brand of music, whilst rooted in rock, retains elements of other genres such as electronica and house music.

 The band was formed at the 1999 Reading Festival by Okereke and Lissack.

 They went through a variety of names before settling on Bloc Party in 2003.

 Moakes joined the band after answering an advert in "NME" magazine, while Tong was picked via an audition.

 Bloc Party got their break by giving BBC Radio 1 DJ Steve Lamacq and Franz Ferdinand's lead singer, Alex Kapranos, a copy of their demo "She's Hearing Voices".
</item 7>

<item 8>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 8>

<item 9>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 9>

<item 10>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 10>
=== END ITEM id=5d1ef74e4a9f3e4c ===

=== ITEM id=5fdfe2934551bbfd ===
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
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 1>

<item 2>
Jerry Beck
Peabody and Sherman", DreamWorks' "Madagascar", and "Bee Movie".

 Beck is also an entertainment industry consultant for TV and home entertainment productions and releases related to classic cartoons and operates the blog
"Cartoon Research."

 He appears frequently as a documentary subject and audio commentator on releases of A&E's "Cartoons Go to War" as well as DVD collections of "Looney Tunes",
"Popeye the Sailor", and "Woody Woodpecker" cartoons, on which he serves as consultant and curator.
</item 2>

<item 3>
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
</item 3>

<item 4>
Ralph Smart
Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in
England to Australian parents.
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
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
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
The entries profile actors, voice performers, puppeteers, composers, writers, and animation history. Ken Barrie voiced and sang for *Postman Pat*; Roger Kitter
played Captain Bertorelli in *’Allo ’Allo!* and scored a tennis-themed chart hit. Ralph Bakshi directed influential adult animations including *Fritz the Cat*,
*Heavy Traffic*, and *Fire and Ice*, plus live-action *Cool and the Crazy*. Other profiles include Disney composer Buddy Baker; Sc
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
Trey Parker
Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.

 He is known for co-creating "South Park" (1997–present) along with his creative partner Matt Stone, as well as co-writing and co-directing the Tony
Award-winning musical "The Book of Mormon" (2011).

 Parker was interested in film and music as a child, and attended the University of Colorado, Boulder following high school, where he met Stone.

 The two collaborated on various short films, and starred in a feature-length musical, titled "Cannibal!

 The Musical" (1993).
</item 10>
=== END ITEM id=5fdfe2934551bbfd ===

=== ITEM id=bd81ad3540f6068b ===
Gold sentences and facts:
Sentence 0: David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
  Fact 0.0: David Ross Lederman was born December 12, 1894.
  Fact 0.1: David Ross Lederman died August 24, 1972.
  Fact 0.2: David Ross Lederman was an American film director.
  Fact 0.3: David Ross Lederman was noted for his Western, action, and adventure films of the 1930s and 1940s.
Sentence 1: Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
  Fact 1.0: Stefan Ruzowitzky was born 25 December 1961.
  Fact 1.1: Stefan Ruzowitzky is an Austrian film director.
  Fact 1.2: Stefan Ruzowitzky is an Austrian screenwriter.

Retrieved items:
<item 1>
Thom Eberhardt
Thom Eberhardt (born March 7, 1947) is an American film director, producer and screenwriter.

 Eberhardt has won two awards and two nominations.

 He is most noted for his work on "Captain Ron", "Honey, I Blew Up the Kid", and the cult classic "Night of the Comet".
</item 1>

<item 2>
Anatomy (film)
Anatomy (German: "Anatomie" ) is a 2000 German horror film written and directed by Stefan Ruzowitzky that stars Franka Potente.

 A sequel, "Anatomy 2" ("Anatomie 2") was released in 2003.

 The film was a box-office success in Germany, and Columbia Pictures released the film's English-dubbed version in the United States theatrically.
</item 2>

<item 3>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 3>

<item 4>
Anatomy 2
Anatomy 2 (German: "Anatomie 2" ) is a 2003 German thriller film written and directed by Stefan Ruzowitzky.

 It's the sequel to the 2000 film "Anatomy", that starred Franka Potente.

 The story moves to Berlin for this film.
</item 4>

<item 5>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 5>

<item 6>
The Counterfeiters (2007 film)
The Counterfeiters (German: "Die Fälscher" ) is a 2007 Austrian-German drama film written and directed by Stefan Ruzowitzky.

 It fictionalizes Operation Bernhard, a secret plan by Nazi Germany during World War II to destabilize the United Kingdom by flooding its economy with forged
Bank of England pound notes.

 The film centres on a Jewish counterfeiter, Salomon 'Sally' Sorowitsch, who is coerced into assisting the operation at the Sachsenhausen concentration camp.
</item 6>

<item 7>
The Inheritors (1998 film)
The Inheritors (original German title "Die Siebtelbauern" – "The Seventh-Part Farmers") is a 1998 Austrian-German film directed by Stefan Ruzowitzky.

 It stars Simon Schwarz and Sophie Rois and has won numerous awards.

 The film was selected as the Austrian entry for the Best Foreign Language Film at the 71st Academy Awards, but was not accepted as a nominee.
</item 7>

<item 8>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 8>

<item 9>
Shadows of the Night (film)
Shadows of the Night is a 1928 American drama silent film directed by D. Ross Lederman and written by Robert E. Hopkins and D. Ross Lederman.

 The film stars Flash the Dog, Lawrence Gray, Louise Lorraine, Warner Richmond and Tom Dugan.

 It was released on October 26, 1928, by Metro-Goldwyn-Mayer.
</item 9>

<item 10>
Javier Frana
Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
</item 10>
=== END ITEM id=bd81ad3540f6068b ===

=== ITEM id=cefffd4f3aa3e7ed ===
Gold sentences and facts:
Sentence 0: Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.
  Fact 0.0: Age of Chance was a British alternative rock-dance crossover band.
  Fact 0.1: Age of Chance was from Leeds, England.
  Fact 0.2: Age of Chance was active from 1985 to 1991.
Sentence 1: Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.
  Fact 1.0: Vector was a rock band.
  Fact 1.1: Vector was formed in Sacramento, California.
  Fact 1.2: Vector was formed in the early 1980s.
  Fact 1.3: Vector was formed by Jimmy Abegg.
  Fact 1.4: Vector was formed by Steve Griffith.
  Fact 1.5: Vector was formed by Charlie Peacock.

Retrieved items:
<item 1>
Binomial pair
In linguistics, a binomial pair or binomial is a sequence of two or more words or phrases belonging to the same grammatical category, having some semantic
relationship and joined by some syntactic device such as "and" or "or".

 Examples in English include "through and through", "(without) let or hindrance", and "chalk and cheese".
</item 1>

<item 2>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 2>

<item 3>
Word2vec
Word2vec is a group of related models that are used to produce word embeddings.

 These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

 Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus
being assigned a corresponding vector in the space.

 Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the
space.
</item 3>

<item 4>
Vector (band)
Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.

 The band had several drummers over the years, including Aaron Smith and Bruce Spencer, both of whom also played drums for The 77s.
</item 4>

<item 5>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 5>

<item 6>
Multivariate random variable
In probability, and statistics, a multivariate random variable or random vector is a list of mathematical variables each of whose value is unknown, either
because the value has not yet occurred or because there is imperfect knowledge of its value.

 The individual variables in a random vector are grouped together because they are all part of a single mathematical system — often they represent different
properties of an individual statistical unit.

 For example, while a given person has a specific age, height and weight, the representation of these features of "an unspecified person" from within a group
would be a random vector.

 Normally each element of a random vector is a real number.
</item 6>

<item 7>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 7>

<item 8>
Doublecross (series)
The 39 Clues: Doublecross is the fourth series in the 39 Clues series of books.

 It follows the story of the Cahills as they suffer from infighting.

 In the first book, Mission Titanic, Ian Kabra has become the new leader of the Cahills.

 Dan and Amy decided to take a break from the Cahills after nearly dying from combating Pierce and his thugs.

 Although Ian believe that he is much more fit to rule the Cahills that Amy and Dan, it seems that the Cahills do not fully recognize Ian as the leader.

 Ian has suspicions that many Cahills are plotting a coup.

 When this coup is realized, Ian Kabra and Cara Pierce are expelled from the Cahill home in Attleboro, Massachusetts.

 The leader of the coup is the Outcast, and he plans to replicate history's worst disasters in order to test Cahill leadership.

 The Cahills are expected to prevent the disasters in order to prove that they are worthy.

 Throughout the series, the Cahills make many shocking discoveries, discovering that Grace was actually ruthless and cruel, and that the Outcast is Nathaniel
Hartford, the husband of Grace whom Grace wanted dead.

 There are four books in the series:
</item 8>

<item 9>
Transstadial transmission
Transstadial transmission occurs when a pathogen remains with the vector from one life stage ("stadium") to the next.

 For example, the bacteria "Borrelia burgdorferi", the causative agent for Lyme disease, infects the tick vector as a larva, and the infection is maintained
when it molts to a nymph and later develops as an adult.

 This type of transmission is seen in other parasites like viruses or "Rickettsia".

 In addition to ticks, mites are another common vector.

 Some sources consider transstadial transmission a type of horizontal transmission, whereas other sources consider it vertical or partial vertical transmission.
</item 9>

<item 10>
New Valley LLC
New Valley LLC is an investment company owned by Vector Group.

 Western Union was renamed New Valley in the early 1990s as the corporate parent of all Western Union related businesses.

 In 1994, New Valley sold its Western Union businesses to First Financial Management (later acquired by First Data) for $1.2 Billion.

 After more than 100 years of being a publicly traded entity, New Valley became a wholly owned subsidiary of Vector Group on December 13, 2005 when Vector Group
acquired the remaining 42.3% of New Valley's common shares that it did not already own.

 New Valley presently has a 70% stake in Douglas Elliman Realty, the largest residential real estate brokerage in the New York metropolitan area.

 The company also has stakes in various hotels and golf courses in the United States.

 In conjunction with the spin-off of its Ladenburg Thalmann subsidiary New Valley retained a 7.6% stake in Ladenburg Thalmann Financial Services, an investment
bank and stock brokerage based in Miami.
</item 10>
=== END ITEM id=cefffd4f3aa3e7ed ===

=== ITEM id=aecf8a85b6b9b7ea ===
Gold sentences and facts:
Sentence 0: Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.
  Fact 0.0: Marcel Émile Verdet was born on 13 March 1824.
  Fact 0.1: Marcel Émile Verdet died on 3 June 1866.
  Fact 0.2: Marcel Émile Verdet was a French physicist.
Sentence 1:  Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de physique".
  Fact 1.0: Marcel Émile Verdet did much to champion the early theory of the conservation of energy in France.
  Fact 1.1: Marcel Émile Verdet championed this theory in France through his editorial supervision of the 'Annales de chimie et de physique'.
Sentence 2: Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the title "Annales de chimie".
  Fact 2.0: Annales de chimie et de physique is French for 'Annals of Chemistry and of Physics'.
  Fact 2.1: Annales de chimie et de physique is a scientific journal.
  Fact 2.2: Annales de chimie et de physique was founded in Paris, France, in 1789.
  Fact 2.3: Annales de chimie et de physique was founded under the title 'Annales de chimie'.

Retrieved items:
<item 1>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 1>

<item 2>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 2>

<item 3>
Index of Middle English Verse
The Index of Middle English Verse (IMEV) is a bibliographic index of poetry in Middle English.

 Its first print publication, in 1943, was an extension of Carleton Brown's "Register of Middle English Religious & Didactic Verse", augmented by the inclusion
of secular verse.

 This edition, edited by Brown and Rossell Hope Robbins, contained entries for over 4000 Middle English poems in more than 2000 manuscripts.

 In 1965 the index was supplemented by Robbins and John L. Cutler.

 The IMEV is available online through Virginia Polytechnic Institute and State University.
</item 3>

<item 4>
The entries cover pornographic publisher Auguste Brancart; Albert Camus, his absurdist philosophy, Nobel Prize, fiction and essays opposing nihilism and capital
punishment; and his biographers. Other topics include the disputed Smiley Face murder theory; Joe McGinniss, Jeffrey MacDonald and the Fatal Vision controversy;
the Guantánamo memoir Enemy Combatant; New Zealand’s Parker–Hulme case and Mazengarb Report; the Lindbergh kidnapping; Nazi collaborator Christian de
</item 4>

<item 5>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 5>

<item 6>
Entente Cordiale (opera)
Entente Cordiale is a comic opera in one act by Ethel Smyth with an English-language libretto by Smyth, who describes the work as "a post-war comedy in one act
(founded on fact)".

 It was first performed by students at the Royal College of Music in London on 22 July 1925.
</item 6>

<item 7>
Villard (imprint)
Villard, also known as Villard Books, is a publishing imprint of Random House, one of the largest publishing companies in the world.

 It was founded in 1983.

 Villard began as an independent imprint of Random House and is currently a sub-imprint of Ballantine Books, itself an imprint of Random House.

 It was named after a Stanford White brownstone mansion on Madison Avenue that was the home of Random House for twenty years.
</item 7>

<item 8>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 8>

<item 9>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 9>

<item 10>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 10>
=== END ITEM id=aecf8a85b6b9b7ea ===

=== ITEM id=7ce7b44ebd85b298 ===
Gold sentences and facts:
Sentence 0: Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.
  Fact 0.0: Jack and Jill is an American magazine for children.
  Fact 0.1: Jack and Jill is published bimonthly.
  Fact 0.2: Jack and Jill is for children 7 to 12 years old.
  Fact 0.3: The title of Jack and Jill comes from the nursery rhyme of the same name.
Sentence 1: Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.
  Fact 1.0: Paper, stylized as PAPER, is a New York City-based magazine.
  Fact 1.1: Paper is an independent magazine.
  Fact 1.2: Paper focuses on fashion.
  Fact 1.3: Paper focuses on popular culture.
  Fact 1.4: Paper focuses on nightlife.
  Fact 1.5: Paper focuses on music.
  Fact 1.6: Paper focuses on art.
  Fact 1.7: Paper focuses on film.

Retrieved items:
<item 1>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 1>

<item 2>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 2>

<item 3>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 3>

<item 4>
Jack and Jill (magazine)
Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.

 It features stories and educational activities.
</item 4>

<item 5>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 5>

<item 6>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 6>

<item 7>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 7>

<item 8>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 8>

<item 9>
Mighty Jill Off
Mighty Jill Off is a 2D platform independently developed freeware video game designed by Anna Anthropy, with art by James Harvey and music by Andrew Toups.

 It stars a submissive named Jill, who has a boot fetish and is forced to climb up a tower after her Queen kicks her down it as punishment.

 Jill does this by jumping and slowly descending over obstacles.

 Jill can be defeated in one hit by these obstacles, but will return to the last check point.

 The game serves as an homage to the 1986 arcade game "Mighty Bomb Jack".

 It had follow-ups, such as "Mighty Jill Off - Jill Off Harder Edition" and "Jill Off With One Hand".

 Jill made a cameo appearance in the 2010 video game "Super Meat Boy" as a playable character.
</item 9>

<item 10>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 10>
=== END ITEM id=7ce7b44ebd85b298 ===

=== ITEM id=edefffc87e75d002 ===
Gold sentences and facts:
Sentence 0: Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
  Fact 0.0: The Spanish name of Teide National Park is 'Parque nacional del Teide'.
  Fact 0.1: Teide National Park is a national park.
  Fact 0.2: Teide National Park is located in Tenerife.
  Fact 0.3: Tenerife is in the Canary Islands, Spain.
Sentence 1: Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands (Spain).
  Fact 1.0: The Spanish name of Garajonay National Park is 'Parque nacional de Garajonay'.
  Fact 1.1: Garajonay National Park is located in the center of the island of La Gomera.
  Fact 1.2: Garajonay National Park is located in the north of the island of La Gomera.
  Fact 1.3: La Gomera is one of the Canary Islands in Spain.

Retrieved items:
<item 1>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 1>

<item 2>
North Cascades National Park
North Cascades National Park is a U.S. National Park located in the state of Washington.

 The park is the largest of the three National Park Service units that comprise the North Cascades National Park Service Complex.

 Several national wilderness areas and British Columbia parkland adjoin the National Park.

 The park features rugged mountain peaks and protects portions of the North Cascades range.
</item 2>

<item 3>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 3>

<item 4>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 4>

<item 5>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 5>

<item 6>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 6>

<item 7>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 7>

<item 8>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 8>

<item 9>
Komodo National Park
Komodo National Park is a national park in Indonesia located within the Lesser Sunda Islands in the border region between the provinces of East Nusa Tenggara
and West Nusa Tenggara.

 The park includes the three larger islands Komodo, Padar and Rinca, and 26 smaller ones, with a total area of 1,733 km (603 km of it land).

 The national park was founded in 1980 to protect the Komodo dragon, the world's largest lizard.

 Later it was dedicated to protecting other species, including marine species.

 In 1991 the national park was declared a UNESCO World Heritage Site.
</item 9>

<item 10>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 10>
=== END ITEM id=edefffc87e75d002 ===

=== ITEM id=1c891c3d62284660 ===
Gold sentences and facts:
Sentence 0: Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.
  Fact 0.0: Cars 2 is a film.
  Fact 0.1: Cars 2 was released in 2011.
  Fact 0.2: Cars 2 is American.
  Fact 0.3: Cars 2 is computer-animated.
  Fact 0.4: Cars 2 is an action comedy spy film.
  Fact 0.5: Cars 2 was produced by Pixar Animation Studios.
  Fact 0.6: Cars 2 was produced for Walt Disney Pictures.
Sentence 1: Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway musicals "Hamilton" and "In the Heights".
  Fact 1.0: Lin-Manuel Miranda was born January 16, 1980.
  Fact 1.1: Lin-Manuel Miranda is American.
  Fact 1.2: Lin-Manuel Miranda is a composer, lyricist, playwright, and actor.
  Fact 1.3: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical Hamilton.
  Fact 1.4: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical In the Heights.
Sentence 2:  He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".
  Fact 2.0: Lin-Manuel Miranda co-wrote the songs for Disney's Moana soundtrack.
  Fact 2.1: The Moana soundtrack is from 2016.
  Fact 2.2: Lin-Manuel Miranda is set to star in Disney's upcoming film Mary Poppins Returns.

Retrieved items:
<item 1>
Cabinet Battle
"Cabinet Battle #1", "Cabinet Battle #2", and "Cabinet Battle #3" are songs written for Act II of the musical "Hamilton", based on the life of Alexander
Hamilton, which premiered on Broadway in 2015.

 Lin-Manuel Miranda wrote both the music and lyrics to the song.
</item 1>

<item 2>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 2>

<item 3>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 3>

<item 4>
Moana (soundtrack)
Moana: Original Motion Picture Soundtrack is the soundtrack to the 2016 Disney animated film "Moana".

 The soundtrack was released by Walt Disney Records on November 19, 2016.

 It features songs written by Lin-Manuel Miranda, Mark Mancina and Opetaia Foa'i, with lyrics in English, Samoan, Tokelauan, and Tuvalu.

 The two-disc deluxe edition includes the score, which was composed by Mancina, as well as demos, outtakes and instrumental karaoke tracks.
</item 4>

<item 5>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 5>

<item 6>
Toy Story 3
Toy Story 3 is a 2010 American 3D computer-animated comedy-drama film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the third installment in Pixar's "Toy Story" series, and the sequel to 1999's "Toy Story 2".

 It was directed by Lee Unkrich, the editor of the first two films and the co-director of "Toy Story 2", written by Michael Arndt, while Unkrich wrote the story
along with John Lasseter and Andrew Stanton, respectively director and co-writer of the first two films.

 The plot focuses on the toys Woody, Buzz Lightyear, and their friends dealing with an uncertain future as their owner, Andy, prepares to leave for college.
</item 6>

<item 7>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 7>

<item 8>
Cars 2 (video game)
Cars 2 (also known as Cars 2: The Video Game) is a racing game developed by Avalanche Software.

 It is based on the film "Cars 2", the sequel to the 2006 film "Cars".

 Originally announced at E3 2011, the game was released by Disney Interactive on major platforms in North America on June 21, 2011, and in Australia 2 days
later.

 The game was released in Europe on July 22, 2011.

 The PlayStation 3 version of the game was reported to feature stereoscopic 3D gameplay.

 A Nintendo 3DS version was released on November 1, 2011 (the same day the film was released on DVD and Blu-ray).

 A PlayStation Portable version was released in North America on November 8, 2011.

 It was also released in Europe on November 4, 2011.

 It was the last Pixar game to be released for the PlayStation Portable.
</item 8>

<item 9>
Cars 2
Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the sequel to 2006's "Cars", and features the voices of Owen Wilson, Larry the Cable Guy, Michael Caine, Emily Mortimer, John Turturro, and Eddie Izzard.

 In the film, race car Lightning McQueen and tow truck Mater head to Japan and Europe to compete in the World Grand Prix, but Mater becomes sidetracked with
international espionage.

 The film was directed by John Lasseter, written by Ben Queen, and produced by Denise Ream.
</item 9>

<item 10>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 10>
=== END ITEM id=1c891c3d62284660 ===

=== ITEM id=f5ef6af72fd97e44 ===
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
Kill Your Heroes
"Kill Your Heroes" is a song by American alternative rock band Awolnation.

 It is written by lead singer Aaron Bruno and musician Brian West for the band's debut studio album "Megalithic Symphony", where it appears as the eighth track.

 "Kill Your Heroes" was released as the third and final single from "Megalithic Symphony" and reached the top 20 of the United States "Billboard" Alternative
Songs and Rock Songs charts.
</item 1>

<item 2>
Awolnation discography
American alternative rock band Awolnation has released two studio albums, three extended plays, eight singles and ten music videos.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.

 The band's second studio album, "Run", was released on March 17, 2015.
</item 2>

<item 3>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 3>

<item 4>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 4>

<item 5>
This Mortal Coil
This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.

 Although Watts-Russell and John Fryer were technically the only two official members, the band's recorded output featured a large rotating cast of supporting
artists, many of whom were otherwise associated with 4AD, including members of Cocteau Twins, Colourbox, and Dead Can Dance.

 The project became known for its gothic, dream pop sound, and released three full albums beginning in 1984 with "It'll End in Tears".
</item 5>

<item 6>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 6>

<item 7>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 7>

<item 8>
Awolnation
Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.
</item 8>

<item 9>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 9>

<item 10>
Hollow Moon (Bad Wolf)
"Hollow Moon (Bad Wolf)" is a song by American alternative rock band Awolnation.

 It was released as the lead single from their second studio album "Run" (2015).

 The song became their first number-one hit on the Billboard Alternative Songs chart in early 2015.
</item 10>
=== END ITEM id=f5ef6af72fd97e44 ===

=== ITEM id=d4e7158cce390689 ===
Gold sentences and facts:
Sentence 0: Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.
  Fact 0.0: Kim Dong-jun's Hangul name is 김동준.
  Fact 0.1: Kim Dong-jun's Hanja name is 金桐俊.
  Fact 0.2: Kim Dong-jun was born February 11, 1992.
  Fact 0.3: Kim Dong-jun is a South Korean singer.
  Fact 0.4: Kim Dong-jun is a South Korean actor.
Sentence 1: Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.
  Fact 1.0: Charlie Harper was born David Charles Perez.
  Fact 1.1: Charlie Harper was born on 25 May 1944.
  Fact 1.2: Charlie Harper was born in Hackney, London.
  Fact 1.3: Charlie Harper is a British singer.
  Fact 1.4: Charlie Harper is a British songwriter.

Retrieved items:
<item 1>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 1>

<item 2>
Kim Dong-jun (footballer)
Kim Dong-jun (Korean: 김동준 , born 19 December 1994) is a South Korean footballer who plays as a goalkeeper for Yonsei University and South Korea national team.
</item 2>

<item 3>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 3>

<item 4>
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
</item 4>

<item 5>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 5>

<item 6>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 6>

<item 7>
Charlie Harper (singer)
Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.

 He is best known as the lead singer from the punk band U.K. Subs.
</item 7>

<item 8>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 8>

<item 9>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 9>

<item 10>
Eunhyuk
Lee Hyuk-jae (born April 4, 1986), better known by his stage name Eunhyuk, is a South Korean singer-songwriter and actor.

 He is a member of the South Korean boy group Super Junior and their subgroups, Super Junior-T and Super Junior-H.

 In 2011, he joined Super Junior's Mandopop subgroup, Super Junior-M and the duo Donghae & Eunhyuk; active in China and Japan respectively.
</item 10>
=== END ITEM id=d4e7158cce390689 ===

=== ITEM id=61d136f0c1b445f1 ===
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
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 1>

<item 2>
Graph query row:
entity: Stiffelio
labels: ['__KGBuilder__', '__Entity__', 'CreativeWork']
type: opera
connected_concept: None
relationship: None
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
List of major opera composers
This list provides a guide to opera composers, as determined by their presence on a majority of compiled lists of significant opera composers.

 (See the "Lists Consulted" section for full details.)

 The composers run from Jacopo Peri, who wrote the first ever opera in late 16th century Italy, to John Adams, one of the leading figures in the contemporary
operatic world.

 The brief accompanying notes offer an explanation as to why each composer has been considered major.

 Also included is a section about major women opera composers, compiled from the same lists.

 For an introduction to operatic history, see opera.

 The organisation of the list is by birthdate.
</item 4>

<item 5>
Graph query row:
entity: Pravda
labels: ['__KGBuilder__', '__Entity__', 'CreativeWork']
type: Soviet newspaper
connected_concept: None
relationship: None
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
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 7>

<item 8>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 8>

<item 9>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 9>

<item 10>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 10>
=== END ITEM id=61d136f0c1b445f1 ===

=== ITEM id=ab817bdba1985f69 ===
Gold sentences and facts:
Sentence 0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie, Michigan.
  Fact 0.0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant.
  Fact 0.1: The Saint Marys Falls Hydropower Plant is a Cloverland Electric Cooperative hydroelectric plant.
  Fact 0.2: The Saint Marys Falls Hydropower Plant is in Sault Ste. Marie, Michigan.
Sentence 1:  Excavation of the power canal began in September 1898 and was completed in June 1902.
  Fact 1.0: Excavation of the Edison Sault Power Canal began in September 1898.
  Fact 1.1: Excavation of the Edison Sault Power Canal was completed in June 1902.
Sentence 2:  The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
  Fact 2.0: The canal was named a Historic Civil Engineering Landmark in 1983.
  Fact 2.1: The hydroelectric complex was named a Historic Civil Engineering Landmark in 1983.
Sentence 3: The Cayuga–Seneca Canal is a canal in New York, United States.
  Fact 3.0: The Cayuga–Seneca Canal is a canal in New York, United States.
Sentence 4:  It is now part of the New York State Canal System.
  Fact 4.0: The Cayuga–Seneca Canal is now part of the New York State Canal System.

Retrieved items:
<item 1>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 1>

<item 2>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 2>

<item 3>
Power canal
A Power Canal refers to a canal used for hydraulic power generation, rather than for transport of watercraft.

 The power canal was a major factor in the Industrial revolution in New England in the 19th century.

 Most early power canals were mill races used mechanically to transfer power directly from falling water to machinery in mill buildings.

 Later, the hydraulic power generated electricity locally for the same mill factories.

 These power canals were often filled in as electricity (transported by power lines) replaced the need for local water power, and road transport needs or city
expansion needs reclaimed the land.

 Some hydraulic power canals were transformed into local electric generators, but most were closed.

 Remains of power canals can be seen in old mill towns and are often protected as historical structures today.
</item 3>

<item 4>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 4>

<item 5>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 5>

<item 6>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 6>

<item 7>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 7>

<item 8>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 8>

<item 9>
Erie Canal Lock 52 Complex
Erie Canal Lock 52 Complex is a national historic district located at Port Byron and Mentz in Cayuga County, New York.

 The district includes two contributing buildings (the Erie House and the blacksmith shop / mule barn); three contributing engineering structures (Erie Canal
Lock 52, culvert, and canal prism of the enlarged Erie Canal); and archaeological sites associated with the canal operations.

 Lock 52 was constructed 1849-1853 as part of the Enlarged Erie Canal program.

 It remained in operation until the rerouting of the canal under the New York State Barge Canal System in 1917.

 The Erie House was built in 1894 and is a two story frame structure that housed a saloon and hotel.
</item 9>

<item 10>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 10>
=== END ITEM id=ab817bdba1985f69 ===
