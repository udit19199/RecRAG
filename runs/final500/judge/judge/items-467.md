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

=== ITEM id=7235000b8ca1b3ae ===
Gold sentences and facts:
Sentence 0:  The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
  Fact 0.0: The first of the books was optioned by Universal Pictures.
  Fact 0.1: The option followed a bidding war over the manuscript.
Sentence 1:  The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film, and the oldest in terms of the overall film market .
  Fact 1.0: The company was founded in 1912.
  Fact 1.1: The company was founded by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour.
  Fact 1.2: The company is the oldest surviving film studio in the United States.
  Fact 1.3: The company is the world's fourth oldest film studio, after Gaumont, Pathé and Nordisk Film.
  Fact 1.4: The company is the oldest film studio in terms of the overall film market.

Retrieved items:
<item 1>
Midnite Movies
Midnite Movies is a line of B movies released first on VHS and later on DVD by MGM Home Entertainment.

 The line was begun by MGM in March 2001 following its acquisition of Orion Pictures, which bought out Filmways, the owner of American International Pictures.

 AIP had a library of B movies from the 1950s and 1960s that were science fiction, horror, and exploitation films.

 The "Midnite Movies" collection is primarily derived from the AIP library (including most of Roger Corman's and Vincent Price's horror movies) but also
included Hammer Film Productions, Amicus Productions, United Artists, and Empire International Pictures movies as well.

 The DVDs were first released as single films but most later releases would be double features on single double-sided discs.

 Sony Pictures Home Entertainment later became owners of the MGM library and continued the "Midnite Movies" line with distributor 20th Century Fox Home
Entertainment.

 All double feature titles released on the 20th Century Fox label were two-disc packages.

 By 2011, no new titles were forthcoming; the previous catalog titles slowly went out of print and the "Midnite Movies" website was taken down.
</item 1>

<item 2>
Bogle-L'Ouverture Publications
Bogle-L’Ouverture Publications (BLP) is a radical London-based publishing company founded in 1968 by Guyanese activists Jessica Huntley (23 February 1927 – 13
October 2013) and Eric Huntley (born 25 September 1929).

 Named in honour of two outstanding liberation fighters in Caribbean history, Toussaint L'Ouverture and Paul Bogle, the company began operating during a period
in the UK when "books by Black authors or written with a sympathetic view of Black people’s history and culture were rare in mainstream bookshops in the UK."

 Alongside New Beacon Books (founded in 1966) and Allison & Busby (founded in 1967), BLP was one of the first black-owned independent publishing companies in
the UK.

 BLP has been described as "a small, unorthodox, self-financing venture that brought a radical perspective to non-fiction, fiction, poetry and children's
books."
</item 2>

<item 3>
Simon Bloom
Simon Bloom is a series of young adult novels by Michael Reisman.

 Set in the fictional town of Lawnville, New Jersey, it is about a boy who finds a book that lets him control the laws of physics.

 The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
</item 3>

<item 4>
Applewood Books
Applewood Books is a book publishing company founded by Phil Zuckerman in 1976.

 They specialize in publishing exacting recreations of historic books, including complex reprints of children's art and pop-up books and other books published
by methods which duplicate antique publishing techniques.

 They have more than 2000 titles in print.

 In recent years, the company has been working to increase the number of reprints it publishes.

 In 2007, the company published over 300 titles.
</item 4>

<item 5>
Sounds True
Sounds True is a multimedia publishing company founded in 1985 by Tami Simon, with the mission of disseminating spiritual wisdom.

 The company is based in Louisville, Colorado, near Boulder, Colorado.

 The company publishes over 800 spoken-word audio and music recordings, books, multimedia learning resources, and online educational programs from those
prominent in the fields of spirituality, psychology, health, and healing, including NY-Times bestselling authors Eckhart Tolle, Pema Chodron, Geneen Roth, Jon
Kabat-Zinn, Clarissa Pinkola Estés, Andrew Weil, Brené Brown, and Caroline Myss.

 The company organizes and hosts an annual event, dedicated to personal growth and spiritual transformation, called The Wake Up Festival, in August of each
year, in Estes Park, Colorado.
</item 5>

<item 6>
Field Enterprises
Field Enterprises was a private holding company founded on August 31, 1944, by Marshall Field III and others whose main asset was the "Chicago Sun".

 That same year the company acquired the book publishers Simon & Schuster and Pocket Books.
</item 6>

<item 7>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 7>

<item 8>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 8>

<item 9>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=7235000b8ca1b3ae ===

=== ITEM id=03aa7eed5c7cd306 ===
Gold sentences and facts:
Sentence 0: Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.
  Fact 0.0: Haringey Heartlands is an urban regeneration project.
  Fact 0.1: Haringey Heartlands is currently ongoing.
  Fact 0.2: Haringey Heartlands is in the London Borough of Haringey.
  Fact 0.3: The London Borough of Haringey is in north London, England.
Sentence 1: The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.
  Fact 1.0: The Mall Wood Green is a large shopping centre.
  Fact 1.1: The Mall Wood Green is a residential complex.
  Fact 1.2: The Mall Wood Green is in Wood Green, north London.

Retrieved items:
<item 1>
Haringey Heartlands
Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.

 The site is located west of Shopping City in Wood Green.

 The aim of the development is "to create a vibrant and attractive new urban quarter which acts as the civic and cultural Heart of Haringey, integrating with
and benefiting wider communities."
</item 1>

<item 2>
White Hart Lane Community Sports Centre
White Hart Lane Community Sports Centre, also known as the New River Stadium is a rugby league and athletics stadium in Wood Green, Haringey, north London,
England that is home to London Skolars rugby league club, Enfield and Haringey Athletic Club, Wood Green Weightlifting Club, Hashtag United F.C. (YouTube based
football club), Haringey Rhinos rugby union club, Haringey Cycling Club, and the Next Level Football League.
</item 2>

<item 3>
Hounsfield Heights/Briar Hill is a Calgary neighbourhood on an escarpment, known for city and mountain views, with Lions Park C-Train station. North London’s
Wood Green features the Mall (still signed “Wood Green Shopping City”), Zone 3 Underground and Alexandra Palace rail stations, and White Hart Lane Community
Sports Centre/New River Stadium, home to several sports clubs. Other entries cover Heartlands High School, the ongoing Haringey Heartlands regeneration,
Haringey Council, and
</item 3>

<item 4>
Wood Green tube station
Wood Green is a London Underground station on the Piccadilly line.

 The station is between Turnpike Lane and Bounds Green stations and is in Travelcard Zone 3.

 It is located at junction of High Road, Wood Green and Lordship Lane.

 It serves Wood Green Shopping City and the nearby Haringey Council administrative complex as well as a densely populated residential area.

 It is also the closest tube station to the Alexandra Palace Venue.
</item 4>

<item 5>
Alexandra Palace railway station
Alexandra Palace railway station (originally named Wood Green and later Wood Green (Alexandra Park)) is in the London Borough of Haringey in north London, and
is in Zone 3.

 The station, and all trains serving it, are operated by Great Northern.

 Trains mostly go to and from Moorgate on weekdays and weekends.
</item 5>

<item 6>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 6>

<item 7>
Wood Green Town F.C.
Wood Green Town Football Club was a football club based in Tottenham in the London Borough of Haringey, England.

 Formed in 1911 as a breakaway from Tufnell Park, they merged with Edmonton (a descendent of Tufnell Park) in 1973 to form Edmonton & Haringey.

 During their history the club were known as Tufnell Spartans, Wood Green, Wood Green Town and Haringey Borough.
</item 7>

<item 8>
Haringey London Borough Council
Haringey London Borough Council is the local authority for the London Borough of Haringey in Greater London, England.

 It is a London borough council, one of 32 in the United Kingdom capital of London.

 Haringey is divided into 19 wards, each electing three councillors.

 Haringey London Borough Council comprises 48 Labour Party councillors, and 9 Liberal Democrats.

 The council was created by the London Government Act 1963 and replaced three local authorities: Hornsey Borough Council, Tottenham Borough Council and Wood
Green Borough Council.

 The next election to the authority will be in 2018.
</item 8>

<item 9>
Heartlands High School
Heartlands High School is a state secondary school located in Wood Green in the London Borough of Haringey.

 It welcomed its first students in September 2010.

 The headteacher is Mr. Simon Garrill.
</item 9>

<item 10>
The Mall Wood Green
The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.

 It is generally still referred to by its former name of Wood Green Shopping City; the signage on the building still uses that name.
</item 10>
=== END ITEM id=03aa7eed5c7cd306 ===

=== ITEM id=615f31acd10b72de ===
Gold sentences and facts:
Sentence 0:  Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.
  Fact 0.0: Most figures are shaped with the body of Mickey Mouse.
  Fact 0.1: Most figures have different themed markings.
  Fact 0.2: Most figures have different themed colors.
  Fact 0.3: Most figures have different themed patterns.
Sentence 1:  He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.
  Fact 1.0: Mickey Mouse was created by Walt Disney.
  Fact 1.1: Mickey Mouse was created by Ub Iwerks.
  Fact 1.2: Mickey Mouse was created at the Walt Disney Studios.
  Fact 1.3: Mickey Mouse was created in 1928.

Retrieved items:
<item 1>
Jerry Beck
Peabody and Sherman", DreamWorks' "Madagascar", and "Bee Movie".

 Beck is also an entertainment industry consultant for TV and home entertainment productions and releases related to classic cartoons and operates the blog
"Cartoon Research."

 He appears frequently as a documentary subject and audio commentator on releases of A&E's "Cartoons Go to War" as well as DVD collections of "Looney Tunes",
"Popeye the Sailor", and "Woody Woodpecker" cartoons, on which he serves as consultant and curator.
</item 1>

<item 2>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 2>

<item 3>
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
</item 3>

<item 4>
Felix the Cat
Felix the Cat is a funny-animal cartoon character created in the silent film era.

 The anthropomorphic black cat with his black body, white eyes, and giant grin, coupled with the surrealism of the situations in which his cartoons place him,
combine to make Felix one of the most recognized cartoon characters in film history.

 Felix was the first character from animation to attain a level of popularity sufficient to draw movie audiences.
</item 4>

<item 5>
Donald Duck
Donald Duck is a cartoon character created in 1934 at Walt Disney Productions.

 Donald is an anthropomorphic white duck with a yellow-orange bill, legs, and feet.

 He typically wears a sailor shirt and cap with a bow tie.

 Donald is most famous for his semi-intelligible speech and his mischievous and temperamental personality.

 Along with his friend Mickey Mouse, Donald is one of the most popular Disney characters and was included in TV Guide's list of the 50 greatest cartoon
characters of all time in 2002.

 He has appeared in more films than any other Disney character, and is the most published comic book character in the world outside of the superhero genre.
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
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 7>

<item 8>
Vinylmation
Vinylmation is a brand of 1.5", 3", and 9" vinyl collectibles sold at Disney theme parks, select Disney Stores, and the online Disney Store.

 The name "Vinylmation" is a combination of the word "Animation" and "Vinyl".

 Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.

 This means that all figures have round mouse ears, whether or not the painted character has them.

 This is accommodated by painting the ears with backgrounds - stars, stripes, rainbows - to match the painted character.

 The series exception is the Park Starz series which does not resemble Mickey at all, and more closely resembles iconic figures from the parks.

 Vinylmation was first introduced in July 2008, and the first figures were introduced in November 2008.
</item 8>

<item 9>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 9>

<item 10>
Mickey Mouse
Mickey Mouse is a funny animal cartoon character and the official mascot of The Walt Disney Company.

 He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.

 An anthropomorphic mouse who typically wears red shorts, large yellow shoes, and white gloves, Mickey has become one of the world's most recognizable
characters.
</item 10>
=== END ITEM id=615f31acd10b72de ===

=== ITEM id=330ad143587febf6 ===
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
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 1>

<item 2>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 2>

<item 3>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 3>

<item 4>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 4>

<item 5>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 5>

<item 6>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 6>

<item 7>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 7>

<item 8>
José Cura
José Cura (born December 5, 1962 in Rosario, Argentina) is an Argentine operatic tenor, conductor, director, scenographer and photographer known for intense and
original interpretations of opera characters, notably "Otello" in Verdi’s "Otello", "Samson" in Saint-Saëns’ "Samson et Dalila", "Canio" in Ruggero
Leoncavallo's "Pagliacci", "Stiffelio" in Giuseppe Verdi's "Stiffelio" and many others.
</item 8>

<item 9>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 9>

<item 10>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 10>
=== END ITEM id=330ad143587febf6 ===

=== ITEM id=07da871443bb1454 ===
Gold sentences and facts:
Sentence 0: Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.
  Fact 0.0: Guitar World is a monthly music magazine.
  Fact 0.1: Guitar World is devoted to guitarists.
  Fact 0.2: Guitar World has been published since July 1980.
Sentence 1: Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.
  Fact 1.0: Golf Digest is a monthly golf magazine.
  Fact 1.1: Golf Digest is published by Condé Nast Publications.
  Fact 1.2: Golf Digest is published in the United States.
Sentence 2:  The magazine started in 1950, and was sold to The New York Times Company in 1969.
  Fact 2.0: The magazine started in 1950.
  Fact 2.1: The magazine was sold to The New York Times Company in 1969.
Sentence 3:  The headquarters of "Golf Digest" is in Des Moines, Iowa.
  Fact 3.0: The headquarters of Golf Digest is in Des Moines, Iowa.

Retrieved items:
<item 1>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 1>

<item 2>
Augusta National Golf Club
Augusta National Golf Club, located in Augusta, Georgia, is one of the most famous golf clubs in the world.

 Founded by Bobby Jones and Clifford Roberts on the site of the former Fruitland (later Fruitlands) Nursery, the course was designed by Jones and Alister
MacKenzie and opened for play in January 1933.

 Since 1934, it has played host to the annual Masters Tournament, one of the four major championships in professional golf, and the only major played each year
at the same course.

 It was the number one ranked course in "Golf Digest"<nowiki>'</nowiki>s 2009 list of America's 100 greatest courses and is currently the number ten ranked
course on "Golfweek Magazine"' s 2011 list of best classic courses in the United States, in terms of course architecture.
</item 2>

<item 3>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 3>

<item 4>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 4>

<item 5>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 5>

<item 6>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 6>

<item 7>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
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
Sheshan Golf Club
Sheshan Golf Club () is the first premier private golf club in Shanghai, China.

 Founded in 2004, the 18 hole golf course plays to a par of 72 and is designed by Nelson and Haworth Design, and spans over 7,266 yards (6,531 meters).

 Every year, the club plays host to the WGC-HSBC Champions, and is currently the only club to host the Championship.

 Sheshan Golf Club was awarded the Best Golf Course in China by "Golf Digest" in 2013 and currently tops of the list of Top Clubs in China by "Golf Magazine".
</item 9>

<item 10>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 10>
=== END ITEM id=07da871443bb1454 ===

=== ITEM id=0319036c95c0de8b ===
Gold sentences and facts:
Sentence 0: Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.
  Fact 0.0: Batman is a 1989 American superhero film.
  Fact 0.1: Batman was directed by Tim Burton.
  Fact 0.2: Batman was produced by Jon Peters and Peter Guber.
  Fact 0.3: Batman is based on the DC Comics character of the same name.
Sentence 1:  He is Chairman and CEO of Mandalay Entertainment.
  Fact 1.0: The person referred to as 'he' is Chairman of Mandalay Entertainment.
  Fact 1.1: The person referred to as 'he' is CEO of Mandalay Entertainment.

Retrieved items:
<item 1>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 1>

<item 2>
Peter Guber
Howard Peter Guber (born March 1, 1942) is an executive, entrepreneur, educator, and author.

 He is Chairman and CEO of Mandalay Entertainment.

 Guber's most recent films from Mandalay Entertainment include "The Kids Are All Right", "Soul Surfer", and "Bernie".

 He has also produced "Batman", "The Witches of Eastwick", and "Flashdance."

 Guber's films have earned over $3 billion worldwide and 50 Academy Award nominations.
</item 2>

<item 3>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 3>

<item 4>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 4>

<item 5>
Batman vs. Two-Face
Batman vs. Two-Face is an upcoming 2017 American animated direct-to-video superhero film produced by Warner Bros.

 Animation and distributed by Warner Bros.

 Home Entertainment.

 It is a sequel to "".

 It will premiere at the New York Comic Con on October 8, 2017 and will be followed by a digital release on October 10, 2017 and on DVD and Blu-ray on October
17, 2017.

 Based on the 1960s "Batman" TV series, the film stars Adam West (in his final role before his death), Burt Ward and Julie Newmar reprising their roles of
Batman, Robin and Catwoman from the series.

 The film will show a tribute to the late Adam West.

 This could also be the final time Warner Bros.

 Animation would make a 1960s Batman film.
</item 5>

<item 6>
Batman: Return of the Joker
Batman: Return of the Joker, known in Japan as Dynamite Batman (ダイナマイトバットマン , Dainamaito Battoman ) , is a 1991 platform video game, the follow-up to Sunsoft's
on the Nintendo Entertainment System.

 Unlike that game, which was based on the 1989 Batman film directed by Tim Burton, "Return of the Joker" is entirely self-contained and based more on the modern
comic book iteration of Batman.

 However, Batman mans the Batmobile from the 1989 film.

 A remake of "Return of the Joker", titled Batman: Revenge of the Joker, was released on the Sega Genesis by Ringler Studios in 1992.

 A Super NES version of "Revenge of the Joker" was in development, but never released.
</item 6>

<item 7>
Batman Forever (disambiguation)
Batman Forever is a 1995 film in the Batman film series.
</item 7>

<item 8>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 8>

<item 9>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 9>

<item 10>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 10>
=== END ITEM id=0319036c95c0de8b ===

=== ITEM id=8b68a8a5c6e79488 ===
Gold sentences and facts:
Sentence 0: Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.
  Fact 0.0: Christopher Edward Nolan was born 30 July 1970
  Fact 0.1: Christopher Edward Nolan is English-American
  Fact 0.2: Christopher Edward Nolan is a film director
  Fact 0.3: Christopher Edward Nolan is a producer
  Fact 0.4: Christopher Edward Nolan is a screenwriter
Sentence 1: Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.
  Fact 1.0: Paul Joseph Schrader was born July 22, 1946
  Fact 1.1: Paul Joseph Schrader is an American screenwriter
  Fact 1.2: Paul Joseph Schrader is a film director
  Fact 1.3: Paul Joseph Schrader is a film critic

Retrieved items:
<item 1>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
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
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 3>

<item 4>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 4>

<item 5>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 5>

<item 6>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 6>

<item 7>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 7>

<item 8>
Blue Collar (film)
Blue Collar is a 1978 American crime drama film directed by Paul Schrader, in his directorial debut.

 It was written by Schrader and his brother Leonard, and stars Richard Pryor, Harvey Keitel and Yaphet Kotto.
</item 8>

<item 9>
The Walker
The Walker is a 2007 American-British drama film written and directed by Paul Schrader.

 It is an independent production and is the latest installment in Schrader's "night workers" series of films, starting with "Taxi Driver" in 1976, followed by
"American Gigolo" in 1980 and "Light Sleeper" in 1992.
</item 9>

<item 10>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 10>
=== END ITEM id=8b68a8a5c6e79488 ===

=== ITEM id=672c596619057181 ===
Gold sentences and facts:
Sentence 0:  All matches were played at HC Davos's home arena, Eisstadion Davos.
  Fact 0.0: All matches were played at the home arena of HC Davos.
  Fact 0.1: The home arena of HC Davos is Eisstadion Davos.
Sentence 1: Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.
  Fact 1.0: Hockey Club Davos is a professional Swiss ice hockey club.
  Fact 1.1: Hockey Club Davos is based in Davos, Switzerland.

Retrieved items:
<item 1>
Fife Flyers
Fife Flyers Ice Hockey Club is the oldest professional ice hockey club in the UK, established in 1938.

 The Flyers play their home games at Fife Ice Arena in Kirkcaldy which has a capacity of just Over 3000 (seated and standing).

 The arena is home to not only Fife Flyers, who play in the Elite Ice Hockey League but also to Kirkcaldy Junior Ice Hockey Club.

 Traditionally many of the Flyers' players have come up through the junior ranks to play at a professional level.
</item 1>

<item 2>
Spengler Cup
The Spengler Cup is an annual invitational ice hockey tournament held in Davos, Switzerland.

 First held in 1923, the Spengler Cup is often cited as the oldest invitational ice hockey tournament in the world.

 The event is hosted by the Swiss team HC Davos and played each year in Davos, Switzerland, between Christmas (December 25) and New Year's Day.

 Currently, all games are held at Vaillant Arena.
</item 2>

<item 3>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 3>

<item 4>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 4>

<item 5>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 5>

<item 6>
2016 Spengler Cup
The 90th Spengler Cup was an ice hockey competition being held in Davos, Switzerland between 26 and 31 December 2016.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 6>

<item 7>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 7>

<item 8>
HC Davos
Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.

 Davos plays in the National League (NL).

 The team is usually a strong force in the league and often bolster their roster with Swiss national team players and players who once played for National
Hockey League teams.

 HC Davos also has hosted the annual Spengler Cup tournament, an invitational competition, since 1923.
</item 8>

<item 9>
2015 Spengler Cup
The 89th Spengler Cup was an ice hockey competition held in Davos, Switzerland between 26 and 31 December 2015.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 9>

<item 10>
Vaillant Arena
Vaillant Arena (Eisstadion Davos) is an arena in Davos, Switzerland.

 It is primarily used for ice hockey and is the home arena of HC Davos.

 It holds 7,080 people, of which 3,280 are seated.

 Every year the Spengler Cup is played in this arena.
</item 10>
=== END ITEM id=672c596619057181 ===

=== ITEM id=4452e3686daea931 ===
Gold sentences and facts:
Sentence 0: Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.
  Fact 0.0: Ghost Rider is a song.
  Fact 0.1: Ghost Rider is a song by the protopunk band Suicide.
  Fact 0.2: Ghost Rider appears on the debut album of the protopunk band Suicide.
Sentence 1:  The song is based on the Marvel Comics character.
  Fact 1.0: The song Ghost Rider is based on the Marvel Comics character.
Sentence 2: Suicide is the debut album from the American rock band Suicide.
  Fact 2.0: The album Suicide is the debut album of the American rock band Suicide.
  Fact 2.1: The band Suicide is an American rock band.
Sentence 3:  It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.
  Fact 3.0: The album Suicide was released in 1977.
  Fact 3.1: The album Suicide was released on Red Star Records.
  Fact 3.2: The album Suicide was produced by Craig Leon.
  Fact 3.3: The album Suicide was produced by Marty Thau.

Retrieved items:
<item 1>
Ghost Rider (Suicide song)
Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.

 The song is based on the Marvel Comics character.
</item 1>

<item 2>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 2>

<item 3>
The Aeroplane Flies High
The Aeroplane Flies High is a five-disc box set released by American alternative rock band The Smashing Pumpkins in 1996.

 It contains expanded versions of the five singles from their album "Mellon Collie and the Infinite Sadness" and also included a 44-page booklet with pictures
and writings by the band's lead singer Billy Corgan, as well as lyrics.

 A limited edition release, the box reached number 42 on the "Billboard" charts, and sold 300,000 units (1.5 million discs in all), generating a platinum disc
for the band.

 Originally intended to be limited to 200,000 copies, Virgin Records produced more after the original run sold out due to overwhelming and unexpected demand.

 The album was remastered in 2013 under the supervision of frontman Billy Corgan and reissued on vinyl and as a CD/DVD box set.
</item 3>

<item 4>
It's Superman!
It's Superman!

 is a novel by Tom De Haven based on the comic book superhero Superman.

 It was released on September 15, 2005 in hardcover and August 29, 2006 in paperback.

 The premise tells the tale of Clark Kent's beginnings into becoming Superman, set in the 1930s, where Clark befriends a wrongly convicted photographer named
Willi Berg, and is then taken from Kansas to Hollywood and finally in New York where he meets Lois Lane, fights Lex Luthor, as he debuts in his superhero
persona.

 Despite the setting, this is not about the Golden Age Superman also known as the Superman of Earth-2; as Perry White, the Daily Planet, Lex Luthor's position
and his trademark powers (including flight) are not part of that alternate Earth.

 Rather, it's a Superman period piece set in the 1930s.
</item 4>

<item 5>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 5>

<item 6>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 6>

<item 7>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 7>

<item 8>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 8>

<item 9>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 9>

<item 10>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 10>
=== END ITEM id=4452e3686daea931 ===

=== ITEM id=37eb9f4ffea2b038 ===
Gold sentences and facts:
Sentence 0: The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy Schaal and Henry Gibson.
  Fact 0.0: The 'Burbs is a 1989 American film.
  Fact 0.1: The 'Burbs is a comedy thriller film.
  Fact 0.2: The 'Burbs was directed by Joe Dante.
  Fact 0.3: The 'Burbs stars Tom Hanks.
  Fact 0.4: The 'Burbs stars Bruce Dern.
  Fact 0.5: The 'Burbs stars Carrie Fisher.
  Fact 0.6: The 'Burbs stars Rick Ducommun.
  Fact 0.7: The 'Burbs stars Corey Feldman.
  Fact 0.8: The 'Burbs stars Wendy Schaal.
  Fact 0.9: The 'Burbs stars Henry Gibson.
Sentence 1: Richard "Rick" Ducommun (July 3, 1952 – June 12, 2015) was a Canadian actor, comedian and writer who appeared in films and television.
  Fact 1.0: Rick Ducommun was born on July 3, 1952.
  Fact 1.1: Rick Ducommun died on June 12, 2015.
  Fact 1.2: Rick Ducommun was a Canadian actor.
  Fact 1.3: Rick Ducommun was a Canadian comedian.
  Fact 1.4: Rick Ducommun was a Canadian writer.
  Fact 1.5: Rick Ducommun appeared in films.
  Fact 1.6: Rick Ducommun appeared in television.

Retrieved items:
<item 1>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 1>

<item 2>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 2>

<item 3>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 3>

<item 4>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 4>

<item 5>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 5>

<item 6>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 6>

<item 7>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 7>

<item 8>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 8>

<item 9>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 9>

<item 10>
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 10>
=== END ITEM id=37eb9f4ffea2b038 ===

=== ITEM id=545ec069901e8616 ===
Gold sentences and facts:
Sentence 0: US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.
  Fact 0.0: US Airways Flight 1549 was an Airbus A320-214.
  Fact 0.1: US Airways Flight 1549 took off from New York City's LaGuardia Airport.
  Fact 0.2: US Airways Flight 1549 took off on January 15, 2009.
  Fact 0.3: US Airways Flight 1549 struck a flock of Canada geese three minutes after takeoff.
  Fact 0.4: The bird strike happened just northeast of the George Washington Bridge.
  Fact 0.5: US Airways Flight 1549 consequently lost all engine power.
Sentence 1:  Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.
  Fact 1.0: Pilots could not reach any airport.
  Fact 1.1: Pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River.
  Fact 1.2: The Hudson River ditching was off Midtown Manhattan.
Sentence 2: Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard survived.
  Fact 2.0: Chesley Burnett "Sully" Sullenberger III was born in 1951.
  Fact 2.1: Chesley Sullenberger is an American retired airline captain.
  Fact 2.2: Chesley Sullenberger is celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan.
  Fact 2.3: The plane was disabled by striking a flock of Canada geese immediately after takeoff.
  Fact 2.4: All 155 people aboard survived.

Retrieved items:
<item 1>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 1>

<item 2>
Chesley Sullenberger
Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways
Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard
survived.

 Sullenberger is an international speaker on airline safety and has helped develop new protocols for airline safety.

 He served as the co-chairman, along with First Officer Jeffrey Skiles, of the EAA's Young Eagles youth introduction-to-aviation program from 2009 to 2013.

 He retired from US Airways after 30 years as a commercial pilot on March 3, 2010.

 In May of the following year, Sullenberger was hired by CBS News as an Aviation and Safety Expert.
</item 2>

<item 3>
Water landing
A water landing is, in the broadest sense, a landing on a body of water.

 Some aircraft such as floatplanes land on water as a matter of course.
</item 3>

<item 4>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 4>

<item 5>
Bird strike
A bird strike—sometimes called birdstrike, bird ingestion (for an engine), bird hit, or bird aircraft strike hazard (BASH)—is a collision between an airborne
animal (usually a bird or bat) and a manmade vehicle, especially an aircraft.

 The term is also used for bird deaths resulting from collisions with structures such as power lines, towers and wind turbines (see Bird-skyscraper collisions
and Towerkill).
</item 5>

<item 6>
US Airways Flight 1549
US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of
Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.

 Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.

 All 155 people aboard were rescued by nearby boats and there were few serious injuries.
</item 6>

<item 7>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 7>

<item 8>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 8>

<item 9>
United Airlines Flight 736
United Airlines Flight 736 was a daily U.S. transcontinental passenger flight operated by United Airlines that crashed on April21, 1958, following a mid-air
collision.

 The aircraft assigned to Flight 736, a Douglas DC-7 airliner carrying 47 persons, was flying at cruise altitude above Clark County, Nevada, en route to a
stopover at Denver, Colorado, when it was struck by a United States Air Force fighter jet crewed by two pilots.

 The collision occurred at 8:30 a.m. in clear weather within a major commercial airway; both aircraft fell out of control from 21000 ft and crashed into
unpopulated desert terrain southwest of Las Vegas, Nevada.
</item 9>

<item 10>
ALM Flight 980
ALM Antillean Airlines Flight 980 was a flight scheduled to fly from John F. Kennedy International Airport in New York City to Princess Juliana International
Airport in St. Maarten, Netherlands Antilles, on 2 May 1970.

 After several unsuccessful landing attempts, the aircraft's fuel was exhausted and it made a forced water landing (ditching) in the Caribbean Sea 48 km (30 mi)
off St. Croix, with 23 fatalities and 40 survivors.

 The accident is one of a small number of intentional water ditchings of jet airliners.
</item 10>
=== END ITEM id=545ec069901e8616 ===
