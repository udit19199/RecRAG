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

=== ITEM id=f8c753a511e2e253 ===
Gold sentences and facts:
Sentence 0: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 0.0: Warrel Dane was born on March 7, 1969.
  Fact 0.1: Warrel Dane is the lead singer of the band Sanctuary.
  Fact 0.2: Sanctuary is an American power metal band.
Sentence 1: Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.
  Fact 1.0: Roy Sætre Khantatat was born on March 12, 1970.
  Fact 1.1: Roy Sætre Khantatat is a Norwegian singer-songwriter.

Retrieved items:
<item 1>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 1>

<item 2>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 2>

<item 3>
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
</item 3>

<item 4>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 4>

<item 5>
Andy Warhol
Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual
art movement known as pop art.

 His works explore the relationship between artistic expression, celebrity culture, and advertising that flourished by the 1960s, and span a variety of media,
including painting, silkscreening, photography, film, and sculpture.

 Some of his best known works include the silkscreen paintings "Campbell's Soup Cans" (1962) and "Marilyn Diptych" (1962), the experimental film "Chelsea Girls"
(1966), and the multimedia events known as the "Exploding Plastic Inevitable" (1966–67).
</item 5>

<item 6>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 6>

<item 7>
Cody B. Ware
Cody Morgan Verdecias (born September 4,1988), better known by his stage name Cody B. Ware is an American Rapper, Songwriter and A&R from Queens, New York who
works on hip-hop and black metal.

 He is a member of the Queens-Based music collective World's Fair.

 Verdecias is a child of interracial parents and he is of half-Afro-Rican and half-Jewish descent.

 According to Verdecias, Bad Rabbits "blessed" him with the stage name Cody B. Ware after Retired Pro Wrestler Koko B. Ware.

 Verdecias says that growing up, his musical influences in hip-hop were Nas, Andre 3000, Big Pun, Khujo Goodie of Goodie Mob and he was also inspired by rock
artists such as Daryl Palumbo of Glassjaw and Jacob Bannon of Converge.
</item 7>

<item 8>
Ricky Wilson (American musician)
Ricky Helton Wilson (March 19, 1953 – October 12, 1985) was an American musician best known as the original guitarist and founding member of rock band the
B-52's.

 Born in Athens, Georgia, Wilson was the brother of fellow member Cindy Wilson.

 The B-52's was founded in 1977, when Ricky, his sister Cindy, Kate Pierson, Keith Strickland and Fred Schneider shared a tropical flaming volcano drink at a
Chinese restaurant and, after an impromptu music session at the home of their friend Owen Scott III, played for the first time at a Valentine's Day party for
friends.

.

 Wilson's unusual guitar tunings were a large contribution to the band's quirky sound .
</item 8>

<item 9>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 9>

<item 10>
Roy Khan
Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.

 Commonly known as Roy Khan or simply Khan, he was the lead vocalist for the symphonic power metal band Kamelot from 1997 to his departure in 2011.

 He co-wrote most of Kamelot's songs with the band's guitarist and founder Thomas Youngblood during his tenure.

 Prior to joining Kamelot, Khan was lead singer for the Norwegian progressive metal band Conception from 1991 until 1997 when they disbanded.
</item 10>
=== END ITEM id=f8c753a511e2e253 ===

=== ITEM id=fc73a4eb655e0985 ===
Gold sentences and facts:
Sentence 0: Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer, musician, director, actor, and comedian who holds dual British and American citizenship.
  Fact 0.0: Christopher Haden-Guest is the 5th Baron Haden-Guest.
  Fact 0.1: Christopher Haden-Guest was born on February 5, 1948.
  Fact 0.2: Christopher Haden-Guest is simply known as Christopher Guest.
  Fact 0.3: Christopher Guest is an English-American screenwriter.
  Fact 0.4: Christopher Guest is an English-American composer.
  Fact 0.5: Christopher Guest is an English-American musician.
  Fact 0.6: Christopher Guest is an English-American director.
  Fact 0.7: Christopher Guest is an English-American actor.
  Fact 0.8: Christopher Guest is an English-American comedian.
  Fact 0.9: Christopher Guest holds dual British and American citizenship.
Sentence 1:  Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.0: Christopher Guest is most widely known in Hollywood for having written his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.1: Christopher Guest is most widely known in Hollywood for having directed his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.2: Christopher Guest is most widely known in Hollywood for having starred in his series of comedy films shot in mock-documentary (mockumentary) style.
Sentence 2:  He also directed dark comedies, children's movies, and historical pictures.
  Fact 2.0: Christopher Guest directed dark comedies.
  Fact 2.1: Christopher Guest directed children's movies.
  Fact 2.2: Christopher Guest directed historical pictures.

Retrieved items:
<item 1>
Welcome, or No Trespassing
Welcome, or No Trespassing (Russian: Добро пожаловать, или Посторонним вход воспрещён , "Dobro pozhalovat, ili Postoronnim vkhod vospreshchyon " ) is a Soviet
movie by Elem Klimov made in 1964.

 It is a satirical comedy about the excessive restrictions that children face during their vacation in a Young Pioneer camp, imposed by their masters.

 Most of the actors are children, while the protagonist is the director Dynin, played by Yevgeniy Yevstigneyev.

 The film was selected to be screened in the Cannes Classics section of the 2015 Cannes Film Festival.
</item 1>

<item 2>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 2>

<item 3>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 3>

<item 4>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 4>

<item 5>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 5>

<item 6>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 6>

<item 7>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 7>

<item 8>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 8>

<item 9>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 9>

<item 10>
Kenny Hotz
Kenneth Joel "Kenny" Hotz is a Canadian comedy writer, producer, director, actor and comedian.

 Hotz is a "South Park" consultant and writer, creator/star of the Comedy Central television show "Kenny vs. Spenny", creator and writer of the FX series
"Testees", and the creator/star of "Kenny Hotz's Triumph of the Will".

 He has received numerous awards for his television work— from the Academy of Canadian Cinema & Television the Gemini Awards and the Canadian Comedy Awards—and
is a multiple-time film festival award-winner.

 He is also an award-winning "Vice" contributor and Gulf war photo-journalist.
</item 10>
=== END ITEM id=fc73a4eb655e0985 ===

=== ITEM id=03075803bb336a8f ===
Gold sentences and facts:
Sentence 0: The Wire Fox Terrier is a breed of dog, one of many terrier breeds.
  Fact 0.0: The Wire Fox Terrier is a breed of dog.
  Fact 0.1: The Wire Fox Terrier is one of many terrier breeds.
Sentence 1: The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.
  Fact 1.0: The West Siberian Laika, also called WSL, is a breed of hunting dog.
  Fact 1.1: The West Siberian Laika, also called WSL, is a breed of spitz type.

Retrieved items:
<item 1>
Signal Circuit of Halleston
Signal Circuit of Halleston was a Wire Fox Terrier and winner of the 1926 Westminster Kennel Club Dog Show.

 This was the 50th Westminster show, and the fourth Fox Terrier to win best in show.
</item 1>

<item 2>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 2>

<item 3>
Wire Fox Terrier
The Wire Fox Terrier is a breed of dog, one of many terrier breeds.

 It is a fox terrier, and although it bears a resemblance to the smooth fox terrier, they are believed to have been developed separately.
</item 3>

<item 4>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 4>

<item 5>
West Siberian Laika
The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.

 Russian publications indicate that the term West Siberian Laika loosely applied to hunting dogs originating with the Mansi and Khanty people in Ural and West
Siberia, but there were no standards or registrations of WSL as such until 1930.

 Then WWll disrupted it for a while, but "systematic breeding with registrations" resumed after the war ended, in 1946.

 This was the time the breed began taking modern shape.

 Before that hunters only knew of Mansi Laika and Khanty Laika.

 In early 1960 many hunters in Ural still preferred the term Mansi Laika, when speaking of West Siberian Laika.

 In Russian language, the term Laika originated from the word "layat" that means to bark.

 The word Laika simply means "barker".

 Any hunting Laika is a bark pointer (pointing at animal of interest by barking and staying with the animal ).

 It is a versatile dog depending on use and environment, but in certain parts of the country they have become more specialized.
</item 5>

<item 6>
Old Jock
Old Jock (1859–1871), was a Fox Terrier famous during the late 19th and early 20th centuries.

 A mostly white dog, he ran briefly with a hunting kennel before becoming a show dog, most notably with a victory at the show which popularised the Fox Terrier.

 His main show rivalry was with a dog named Tartar, and along with a dog named Trap, the three were popular sires of the Fox Terrier breed.

 He was also involved in the early formation of the Jack Russell Terrier and the Dandie Dinmont Terrier breeds.
</item 6>

<item 7>
Teddy Roosevelt Terrier
The Teddy Roosevelt Terrier is a small to medium-sized American hunting terrier.

 Lower-set with shorter legs, more muscular, and heavier bone density than its cousin the American Rat Terrier.

 There is much diversity in the history of the Teddy Roosevelt Terrier breed and it shares a common early history with the American Rat Terrier, Fox Paulistinha
and Tenterfield Terrier.

 It is said the Rat Terrier background stems from the terriers or other dogs that were brought over by early English and other working class immigrants.

 Since the breed was a farm, hunting and utility dog there was little to no planned breeding other than breeding dogs with agreeable traits to each other in
order to produce the desired work ethic in the dog.

 It is assumed that the Feist (dog), Bull Terrier, Smooth Fox Terrier, Manchester Terrier, Whippet, Italian Greyhound, the now extinct English White Terrier,
Turnspit dog and or Wry Legged Terrier all share in the Teddy Roosevelt Terrier's ancestry.

 These early Ratting Terriers were then most likely bred to the Beagle or Beagle cross bred dogs (for increased scenting ability) and other dogs.
</item 7>

<item 8>
Pomeranian (dog)
The Pomeranian (often known as a Pom or Pom Pom) is a breed of dog of the Spitz type that is named for the Pomerania region in Germany and Poland in Central
Europe.

 Classed as a toy dog breed because of its small size, the Pomeranian is descended from the larger Spitz type dogs, specifically the German Spitz.

 It has been determined by the Fédération Cynologique Internationale to be part of the German Spitz breed; and in many countries, they are known as the
Zwergspitz ("Dwarf-Spitz").
</item 8>

<item 9>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 9>

<item 10>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 10>
=== END ITEM id=03075803bb336a8f ===

=== ITEM id=3e14d76101a280fc ===
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
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 1>

<item 2>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 2>

<item 3>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 3>

<item 4>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 4>

<item 5>
Dead Poets Society
Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.

 Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students
through his teaching of poetry.
</item 5>

<item 6>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 6>

<item 7>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 7>

<item 8>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 8>

<item 9>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 9>

<item 10>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 10>
=== END ITEM id=3e14d76101a280fc ===

=== ITEM id=44d0a11b4c020376 ===
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
Graph query row:
entity: This Mortal Coil
labels: ['__KGBuilder__', '__Entity__', 'Organization']
kind: music collective
genre: None
nationality: None
relationship: MEMBER_OF
related: Ivo Watts-Russell
related_kind: None
related_genre: None
related_nationality: None
status: None
</item 1>

<item 2>
Graph query row:
entity: Awolnation
labels: ['__KGBuilder__', '__Entity__', 'Organization']
kind: American alternative rock band
genre: alternative rock
nationality: American
relationship: MEMBER_OF
related: Aaron Bruno
related_kind: None
related_genre: None
related_nationality: None
status: None
</item 2>

<item 3>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 3>

<item 4>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 4>

<item 5>
This Mortal Coil
This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.

 Although Watts-Russell and John Fryer were technically the only two official members, the band's recorded output featured a large rotating cast of supporting
artists, many of whom were otherwise associated with 4AD, including members of Cocteau Twins, Colourbox, and Dead Can Dance.

 The project became known for its gothic, dream pop sound, and released three full albums beginning in 1984 with "It'll End in Tears".
</item 5>

<item 6>
Hollow Moon (Bad Wolf)
"Hollow Moon (Bad Wolf)" is a song by American alternative rock band Awolnation.

 It was released as the lead single from their second studio album "Run" (2015).

 The song became their first number-one hit on the Billboard Alternative Songs chart in early 2015.
</item 6>

<item 7>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
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
Kill Your Heroes
"Kill Your Heroes" is a song by American alternative rock band Awolnation.

 It is written by lead singer Aaron Bruno and musician Brian West for the band's debut studio album "Megalithic Symphony", where it appears as the eighth track.

 "Kill Your Heroes" was released as the third and final single from "Megalithic Symphony" and reached the top 20 of the United States "Billboard" Alternative
Songs and Rock Songs charts.
</item 9>

<item 10>
Awolnation discography
American alternative rock band Awolnation has released two studio albums, three extended plays, eight singles and ten music videos.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.

 The band's second studio album, "Run", was released on March 17, 2015.
</item 10>
=== END ITEM id=44d0a11b4c020376 ===

=== ITEM id=8e8a17ac395e6d51 ===
Gold sentences and facts:
Sentence 0: The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.
  Fact 0.0: The Night of the Arts is an annual event.
  Fact 0.1: The Night of the Arts is known in Finnish as 'Taiteiden yö'.
  Fact 0.2: The Night of the Arts is known in Swedish as 'Konstens natt'.
  Fact 0.3: The Night of the Arts is held in several major cities in Finland.
  Fact 0.4: The Night of the Arts is held in late August.
Sentence 1:  It is one of many White Night festivals held worldwide.
  Fact 1.0: The Night of the Arts is one of many White Night festivals.
  Fact 1.1: White Night festivals are held worldwide.
Sentence 2: The White Nights are a kind of all-night arts festival held in many cities in the summer.
  Fact 2.0: The White Nights are a kind of all-night arts festival.
  Fact 2.1: The White Nights festival is held in many cities.
  Fact 2.2: The White Nights festival is held in the summer.
Sentence 3:  The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are early and darkness is never complete.
  Fact 3.0: 'White nights' is the name given in areas of high latitude to the weeks around the summer solstice in June.
  Fact 3.1: During white nights, sunsets are late.
  Fact 3.2: During white nights, sunrises are early.
  Fact 3.3: During white nights, darkness is never complete.

Retrieved items:
<item 1>
Night of the Arts
The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.

 It is one of many White Night festivals held worldwide.
</item 1>

<item 2>
Green Man Festival
Green Man is an independent music and arts festival held annually in mid-August in the Brecon Beacons, Wales since 2003.

 It has evolved into a 20,000 capacity 4-day event, showcasing predominantly live music (in particular alternative, indie, rock, folk, dance and americana),
with additional events showcasing literature, film, comedy, theatre and poetry.

 2015 saw 1,500 multi-arts acts perform across 17 stages.

 The festival site is divided into 10 areas, each offering a unique festival experience.

 Ceilidhs, all-night bonfires and secret gigs all add to the festival's unique identity.
</item 2>

<item 3>
Artscape (festival)
Artscape is an annual art festival held in the Mount Royal neighborhood of Baltimore, Maryland in July.

 Since its first annual event in 1982, it has become the largest free arts festival in America.

 It has boasted acts such as Ray Charles, Aretha Franklin, and Matisyahu in the past, attracting over 350,000 people from the city, and surrounding areas.

 Film programming during Artscape is provided by Maryland Film Festival.

 There are artists in a variety of visual and performing media.

 Events are free and open to the public.
</item 3>

<item 4>
Satumaa
"Satumaa" (roughly, in English "The Fabled Land" or "The Fairytale Land") is the quintessential Finnish tango.

 It was written by Unto Mononen, and published in 1955.

 The most famous recording is probably the one made by Reijo Taipale in 1962.

 The lyrics tell a story of a distant land beyond the sea — a happy paradise — however, the narrator can only reach it in his thoughts.

 The song has been recorded countless times, mainly by male Finnish tango singers.
</item 4>

<item 5>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 5>

<item 6>
White Night festivals
The White Nights are a kind of all-night arts festival held in many cities in the summer.

 The original festival is the White Nights Festival held in Saint Petersburg, Russia.

 The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are
early and darkness is never complete.

 In Saint Petersburg, the Sun does not set until after 10 p.m., and the twilight lasts almost all night.
</item 6>

<item 7>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 7>

<item 8>
Great Northern Arts Festival
The Great Northern Arts Festival is held each year in Inuvik, Northwest Territories.

 Lasting approximately ten days, it is an annual event held each summer.

 It was established in 1989 by Sharlene Alexander and Sue Rose.

 While the festival's mandate is to serve Northern artists and craftspeople, artists from around the world also attend.

 The festival is designed as a venue for artists to see each other's work and to experiment with new applications.

 It includes arts of all types, such as fiber art and Music of the Northwest Territories.

 Artists partake in demonstrations, masterclasses, seminars, and workshops.

 The festival's gallery offers pieces for sale.

 There are cultural presentations each evening which include concerts, dance, fashion, music, and storytelling.

 Media coverage brings festival highlights to viewing audiences.
</item 8>

<item 9>
Kerala School Kalolsavam
School Arts Festival of Kerala (Malayalam: കേരള സ്കൂൾ കലോത്സവം) is an annual event conducted by the state government of Kerala, featuring several art
competitions for high school and higher secondary school students of Kerala.

 The festival was started in 1956, and till 2008, it was called as "Kerala State School Youth Festival".

 The participants are students from classes 8th to 12th.

 Winners from different revenue districts for a particular event will be competing in state level competition.

 The event is usually conducted in December–January months of a year and is considered to be the biggest cultural event of Asia.

 57th edition of Kerala School Kalolsavam 2017 is scheduled for January 16 to 22 at Kannur.
</item 9>

<item 10>
Tanglewood Music and Arts Festival
Tanglewood Music and Arts Festival is an annual two-day music, comedy and arts event held within the grounds of Narrow Water Castle, County Down, Northern
Ireland.

 The festival, founded in 2011, has attracted significant attention both within the local area and around Northern Ireland with buses coming from as far as
Belfast to the event.

 For the 2011 and 2012 festivals Tanglewood was held on one day; however for 2013 it has expanded into a two-day camping festival.
</item 10>
=== END ITEM id=8e8a17ac395e6d51 ===

=== ITEM id=270c4542c0ab9fb3 ===
Gold sentences and facts:
Sentence 0:  He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
  Fact 0.0: The person referred to as 'He' has had a cameo acting role as a tailor in The Wolf of Wall Street.
  Fact 0.1: The person referred to as 'He' has had a cameo acting role as a tailor in The Good Shepherd.
Sentence 1: The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir of the same name by Jordan Belfort.
  Fact 1.0: The Wolf of Wall Street is a 2013 film.
  Fact 1.1: The Wolf of Wall Street is American.
  Fact 1.2: The Wolf of Wall Street is a biographical black comedy crime film.
  Fact 1.3: The Wolf of Wall Street was directed by Martin Scorsese.
  Fact 1.4: The Wolf of Wall Street was written by Terence Winter.
  Fact 1.5: The Wolf of Wall Street is based on the memoir of the same name by Jordan Belfort.

Retrieved items:
<item 1>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 1>

<item 2>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 2>

<item 3>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 3>

<item 4>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 4>

<item 5>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 5>

<item 6>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 6>

<item 7>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 7>

<item 8>
A Walk in the Woods (film)
A Walk in the Woods is a 2015 American biographical comedy-drama film directed by Ken Kwapis, based on the 1998 book/memoir of the same name by Bill Bryson.

 The film stars Robert Redford, Nick Nolte and Emma Thompson.

 The film was released on September 2, 2015, by Broad Green Pictures.
</item 8>

<item 9>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 9>

<item 10>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 10>
=== END ITEM id=270c4542c0ab9fb3 ===

=== ITEM id=86c05fca3ed75826 ===
Gold sentences and facts:
Sentence 0:  Jan Kodeš and Margaret Court won the singles titles.
  Fact 0.0: Jan Kodeš won the singles titles.
  Fact 0.1: Margaret Court won the singles titles.
Sentence 1:  She is currently a Christian minister in Perth, Western Australia.
  Fact 1.0: A woman referred to in this sentence is currently a Christian minister.
  Fact 1.1: The Christian minister referred to in this sentence is based in Perth, Western Australia.

Retrieved items:
<item 1>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 1>

<item 2>
In 2009, Roger Federer won the French Open and Wimbledon, completed the Career Grand Slam, and claimed a record 15th major; he also won Masters titles in Madrid
and Cincinnati. Hicham Arazi, a Moroccan player ranked as high as No. 22, won Casablanca and reached four major quarterfinals. Swiss Marc Rosset won Olympic
singles gold in 1992, the French Open doubles title, and the 2000 AXA Cup. Nag
</item 2>

<item 3>
1972 French Open – Men's Singles
Jan Kodes was the two-time defending champion, but lost in the quarterfinals to Patrick Proisy.
</item 3>

<item 4>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 4>

<item 5>
At the 1976 Italian Open, Adriano Panatta and Mima Jaušovec won singles; Panatta survived 11 match points in round one. Jan Kodeš and Margaret Court won the
1970 French Open. Vitas Gerulaitis won Wimbledon doubles (1975), the 1977 Australian Open, Italian Opens (1977, 1979), and 1978 WCT Finals; he also won Brussels
(1982), Ocean City (1977
</item 5>

<item 6>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 6>

<item 7>
The material compiles tennis results and player biographies. Sherwood Stewart and Ferdi Taygan won the 1982 French Open doubles and finished runner-up at the
Volvo Masters and U.S. Pro Indoor; Stewart later partnered others. Jorge Lozano, a Mexican doubles No. 4, won nine doubles titles, including two French Open
mixed doubles crowns, and captained Mexico’s Davis Cup team. Brazilian Carlos Fernandes reached the 1961 French Open quarterfinals. Argentine Javier Fr
</item 7>

<item 8>
Carlos Fernandes (tennis)
At Roland Garros 1962, Fernandes lost to Lundquist in round two.

 At Wimbledon he lost in round one to Neale Fraser's brother John.

 At 1963 French Open he lost in round one to Christian Duxin.

 In round one of Wimbledon he led a young Arthur Ashe by two sets to 0, but lost in five sets.

 At Wimbledon 1964 he lost in round two to Gene Scott.

 An ankle injury in 1964 took its' toll and Fernandes played less after that.

 As Open tennis arrived, Fernandes was reaching the end of his career.

 He lost in round one of 1968 French Open to Bernard Montrenaud.

 At Wimbledon he lost in round one to Mike Sangster.
</item 8>

<item 9>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 9>

<item 10>
1970 French Open
The 1970 French Open was a tennis tournament that was held at the Stade Roland Garros in Paris in France from 25 May through 7 June 1970.

 It was the 69th edition of the French Open, the 40th to be open to foreign competitors, and the second Grand Slam of the year.

 Jan Kodeš and Margaret Court won the singles titles.
</item 10>
=== END ITEM id=86c05fca3ed75826 ===

=== ITEM id=db8fc358f74cf9a6 ===
Gold sentences and facts:
Sentence 0: The University of Delaware (colloquially "UD") is the largest university in Delaware.
  Fact 0.0: The University of Delaware is colloquially called 'UD'.
  Fact 0.1: The University of Delaware is the largest university in Delaware.
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen's Hungarian name is 'Debreceni Egyetem'.
  Fact 1.1: The University of Debrecen is a university.
  Fact 1.2: The University of Debrecen is located in Debrecen, Hungary.

Retrieved items:
<item 1>
Ellendale, Delaware
Ellendale is a town in Sussex County, Delaware, United States.

 The population was 381 at the 2010 census, an increase of 16.5% since 2000.

 It is part of the Salisbury, Maryland-Delaware Metropolitan Statistical Area.

 Ellendale is the "Gateway to Delaware's Resort Beaches" because it is the town located on U.S. Highway 113, the resort area's westernmost border, and Delaware
Route 16, the resort area's northernmost border with the eastern border being the Delaware Bay and Atlantic Ocean and the southern border being the state line
with Maryland.

 Ellendale is home to the Philadelphia Bible College and the Harbor Christian Academy.
</item 1>

<item 2>
Delaware–William &amp; Mary football rivalry
The Delaware–William & Mary football rivalry between the Delaware Fightin' Blue Hens and the William & Mary Tribe is a match-up between two public universities,
the University of Delaware and the College of William and Mary, that are also members of the Colonial Athletic Association.

 Both schools have academic reputations that have labeled them as Public Ivies.

 Both schools are also Colonial Colleges having been founded before the United States became independent in 1776; the College of William and Mary was founded in
1693 and the University of Delaware's predecessor school was founded in 1743.
</item 2>

<item 3>
University of Delaware Botanic Gardens
The University of Delaware Botanic Gardens are botanical gardens and an arboretum located on the campus of the University of Delaware, in Newark, Delaware,
United States.

 The gardens are open to the public without charge.
</item 3>

<item 4>
Pike Creek, Delaware
Pike Creek is a census-designated place (CDP) in New Castle County, Delaware and is part of the Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metropolitan
statistical area.

 In 2007, CNN's Money Magazine ranked Pike Creek on its list of the 100 Best Places to Live in the United States.

 Pike Creek was the only community in Delaware to appear on the list.

 The population was 7,898 at the 2010 census.

 Local attractions include Middle Run Natural Area, White Clay Creek State Park, Carousel Farm Park & Equestrian Center, Golf and Restaurants.

 Pike Creek is approximately 5 miles from downtown Newark, Delaware which contains the University of Delaware.

 Pike Creek is connected to the University of Delaware campus and the Newark, Delaware downtown area by several bike trails.

 Pike Creek is approximately 10 miles west of the state's largest city, Wilmington, Delaware which is a national financial center.

 Chester County, Pennsylvania is located approximately 2 miles west of Pike Creek.
</item 4>

<item 5>
College Park, Delaware
College Park is an unincorporated community in New Castle County, Delaware, United States.

 College Park is located along Delaware Route 896 southwest of Newark.
</item 5>

<item 6>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 6>

<item 7>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 7>

<item 8>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 8>

<item 9>
Rutgers University–Camden
Rutgers University–Camden is one of three regional campuses of Rutgers University, New Jersey's public research university.

 It is located in Camden, New Jersey, United States.

 Founded in the 1920s, Rutgers–Camden began as an amalgam of the South Jersey Law School and the College of South Jersey.

 It is the southernmost of the three regional campuses of Rutgers—the others being located in New Brunswick and Newark.

 The city of Camden is located on the Delaware River, east of Philadelphia.
</item 9>

<item 10>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 10>
=== END ITEM id=db8fc358f74cf9a6 ===

=== ITEM id=57567eaf5aeaa1ff ===
Gold sentences and facts:
Sentence 0: Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the Himalayas, and Indonesia) and the Americas.
  Fact 0.0: Hydrangea is a genus of flowering plants.
  Fact 0.1: Hydrangea's common names are hydrangea or hortensia.
  Fact 0.2: The Hydrangea genus has 70 to 75 species.
  Fact 0.3: Hydrangea plants are native to southern and eastern Asia.
  Fact 0.4: Hydrangea plants are native to China, Japan, Korea, the Himalayas, and Indonesia.
  Fact 0.5: Hydrangea plants are native to the Americas.
Sentence 1:  It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").
  Fact 1.0: The subject of this sentence contains 13 to 18 species.
  Fact 1.1: The subject of this sentence is best known for a single species.
  Fact 1.2: That single species is Jequirity.
  Fact 1.3: Jequirity is also called A. precatorius.

Retrieved items:
<item 1>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 1>

<item 2>
Hydrangea heteromalla
Hydrangea heteromalla is a species of "Hydrangea" flower, native to the Himalayas and China.

 Average height is 10'-15' with inflorescences that are 5"-8" across and bloom May-June,full flowering only occurs in full sun.Is a very adaptable plant that is
hardy to USDA zones (4)5-7.

 This plant is seldom seen in American gardens although there is nice specimen at the Hillier Arboretum.
</item 2>

<item 3>
Hydrangea serrata
Hydrangea serrata is a species of flowering plant in the family Hydrangeaceae, native to mountainous regions of Korea and Japan.

 Common names include mountain hydrangea and tea of heaven.

 Growing to 1.2 m tall and broad, it is a deciduous shrub with oval leaves and panicles of blue and pink flowers in summer and autumn (fall).

 It is widely cultivated as an attractive ornamental shrub throughout the world in areas with suitable climate and soil.
</item 3>

<item 4>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 4>

<item 5>
Hydrangea paniculata
Hydrangea paniculata, the panicled hydrangea, is a species of flowering plant in the family Hydrangeaceae native to southern and eastern China, Korea, Japan and
Russia (Sakhalin).

 It was first formally described by Philipp Franz von Siebold in 1829.

 It is a deciduous shrub or small tree, 1 - tall by 2.5 m broad, growing in sparse forests or thickets in valleys or on mountain slopes.
</item 5>

<item 6>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 6>

<item 7>
Hydrangea macrophylla
Hydrangea macrophylla is a species of flowering plant in the family Hydrangeaceae, native to Japan.

 It is a deciduous shrub growing to 2 m tall by 2.5 m broad with large heads of pink or blue flowers in summer and autumn.

 Common names include bigleaf hydrangea, French hydrangea, lacecap hydrangea, mophead hydrangea, penny mac and hortensia.

 It is widely cultivated in many parts of the world in many climates.

 It is not to be confused with "H. aspera" 'Macrophylla'.
</item 7>

<item 8>
Abrus canescens
Abrus canescens is a species of flowering plant belonging to the legume family, native to Africa.

 It is considered almost extinct.
</item 8>

<item 9>
Hydrangea arborescens
Hydrangea arborescens, commonly known as smooth hydrangea, wild hydrangea, or sevenbark, is a species of flowering plant in the family Hydrangeaceae.

 It is a small- to medium-sized, deciduous shrub up to 3 m tall that is native to the eastern United States.
</item 9>

<item 10>
Hydrangea quercifolia
Hydrangea quercifolia, commonly known by its translation oakleaf hydrangea or oak-leaved hydrangea, is a species of flowering plant native to the Southeastern
United States, in woodland habitats from North Carolina west to Tennessee, and south to Florida and Louisiana.

 A deciduous shrub with white showy flower heads, it is a commonly grown garden plant.

 Numerous cultivars are available commercially.
</item 10>
=== END ITEM id=57567eaf5aeaa1ff ===

=== ITEM id=5b0d5b47009bf7ce ===
Gold sentences and facts:
Sentence 0: Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.
  Fact 0.0: The Cat in the Hat is a Dr. Seuss film.
  Fact 0.1: The Cat in the Hat is a 2003 film.
  Fact 0.2: The Cat in the Hat is an American film.
  Fact 0.3: The Cat in the Hat is a family comedy film.
  Fact 0.4: The Cat in the Hat was directed by Bo Welch.
Sentence 1: Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
  Fact 1.0: Robert W. 'Bo' Welch III was born November 30, 1951.
  Fact 1.1: Robert W. 'Bo' Welch III is an American production designer.
  Fact 1.2: Robert W. 'Bo' Welch III is an American director.

Retrieved items:
<item 1>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 1>

<item 2>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 2>

<item 3>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 3>

<item 4>
Nitish Roy
Nitish Roy is an Indian film art director, production designer, and costume designer in Hindi cinema and a Bengali Film Director, who is also known for his work
with art cinema directors, Shyam Benegal, Mrinal Sen and Govind Nihlani, Hindi mainstream cinema, directors like Rajkumar Santoshi, as well as international
directors like Mira Nair and Gurinder Chadha.
</item 4>

<item 5>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 5>

<item 6>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 6>

<item 7>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 7>

<item 8>
The Cat in the Hat (film)
Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.

 It is based on the 1957 Dr. Seuss book "of the same name".

 The film stars Mike Myers in the title role of the Cat in the Hat, and Dakota Fanning as Sally.

 Sally's brother (who is unnamed in the book and the 1971 TV special), Conrad, is portrayed by Spencer Breslin.

 The film is the second feature-length Dr. Seuss adaptation after the 2000 holiday film "How the Grinch Stole Christmas".
</item 8>

<item 9>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
</item 9>

<item 10>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 10>
=== END ITEM id=5b0d5b47009bf7ce ===
