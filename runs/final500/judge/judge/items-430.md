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

=== ITEM id=76e91901f24bfe30 ===
Gold sentences and facts:
Sentence 0: A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.
  Fact 0.0: A Summer in the Cage is a documentary film.
  Fact 0.1: A Summer in the Cage was released in 2007.
  Fact 0.2: A Summer in the Cage is about a man's experiences with bipolar disorder.
Sentence 1: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.
  Fact 1.0: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary.
  Fact 1.1: American Hardcore: The History of American Punk Rock 1980-1986 was directed by Paul Rachman.
  Fact 1.2: American Hardcore: The History of American Punk Rock 1980-1986 was written by Steven Blush.

Retrieved items:
<item 1>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 1>

<item 2>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 2>

<item 3>
The entries span film, television, documentaries, and film distribution: Philippine series Sine Novela adapts Viva classics; The Doom Generation (1995) is Gregg
Araki’s violent “Teenage Apocalypse” film; Naked Gun 33⅓ concludes its comedy trilogy, while Naked City (1958–63) is a semi-documentary police drama. Night Life
is a zombie comedy; The Runaways portrays the band’s formation and Currie-Jett relationship. Consolidated Film Industries was
</item 3>

<item 4>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 4>

<item 5>
American Pie Presents: Band Camp
American Pie Presents: Band Camp (also known as American Pie: Band Camp) is a 2005 direct-to-DVD sex comedy film.

 It is the first installment in the "American Pie Presents" series and the fourth installment in the "American Pie" franchise.

 It was directed by Steve Rash and written by Brad Riddell.

 The film centers on Matt Stifler, younger brother of Steve Stifler.

 In this film, Matt is sent to the infamous band camp, where he is compelled to change his ways.
</item 5>

<item 6>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
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
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 8>

<item 9>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 9>

<item 10>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 10>
=== END ITEM id=76e91901f24bfe30 ===

=== ITEM id=d634d670954fe57a ===
Gold sentences and facts:
Sentence 0: The Wire Fox Terrier is a breed of dog, one of many terrier breeds.
  Fact 0.0: The Wire Fox Terrier is a breed of dog.
  Fact 0.1: The Wire Fox Terrier is one of many terrier breeds.
Sentence 1: The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.
  Fact 1.0: The West Siberian Laika, also called WSL, is a breed of hunting dog.
  Fact 1.1: The West Siberian Laika, also called WSL, is a breed of spitz type.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
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
Signal Circuit of Halleston
Signal Circuit of Halleston was a Wire Fox Terrier and winner of the 1926 Westminster Kennel Club Dog Show.

 This was the 50th Westminster show, and the fourth Fox Terrier to win best in show.
</item 7>

<item 8>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 8>

<item 9>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 9>

<item 10>
Pomeranian (dog)
The Pomeranian (often known as a Pom or Pom Pom) is a breed of dog of the Spitz type that is named for the Pomerania region in Germany and Poland in Central
Europe.

 Classed as a toy dog breed because of its small size, the Pomeranian is descended from the larger Spitz type dogs, specifically the German Spitz.

 It has been determined by the Fédération Cynologique Internationale to be part of the German Spitz breed; and in many countries, they are known as the
Zwergspitz ("Dwarf-Spitz").
</item 10>
=== END ITEM id=d634d670954fe57a ===

=== ITEM id=5ec43833e3202226 ===
Gold sentences and facts:
Sentence 0: Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.
  Fact 0.0: Rick Ray is an American filmmaker.
  Fact 0.1: Rick Ray is best known for his 2006 documentary film 10 Questions for the Dalai Lama.
  Fact 0.2: Rick Ray wrote, filmed and directed 10 Questions for the Dalai Lama.
Sentence 1: John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.
  Fact 1.0: John Alan Lasseter was born January 12, 1957.
  Fact 1.1: John Alan Lasseter is an American animator.
  Fact 1.2: John Alan Lasseter is an American film director.
  Fact 1.3: John Alan Lasseter is an American screenwriter.
  Fact 1.4: John Alan Lasseter is an American film producer.
Sentence 2:  He is also the Principal Creative Advisor for Walt Disney Imagineering.
  Fact 2.0: John Alan Lasseter is also the Principal Creative Advisor for Walt Disney Imagineering.

Retrieved items:
<item 1>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 1>

<item 2>
Luxo Jr.
Luxo Jr. is a 1986 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The two-minute short film revolves around one larger and one smaller desk lamp.

 The larger lamp, named Luxo Sr., looks on while the smaller, "younger" Luxo Jr.

 plays exuberantly with a ball that it accidentally deflates.

 "Luxo Jr." was Pixar's first animation after Ed Catmull and John Lasseter left Industrial Light and Magic's computer division.

 It is the source of the hopping desk lamp included in Pixar's corporate logo.
</item 2>

<item 3>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 3>

<item 4>
Grizzly River Run
Grizzly River Run is located at Disney California Adventure at the Disneyland Resort in Anaheim, California.

 It is similar to Kali River Rapids in Disney's Animal Kingdom but distinctive as the rafts are engineered to spin as they descend chutes.

 The attraction's name comes from Grizzly Peak, the bear shaped mountain that the rapids flow around.

 It was designed by Walt Disney Imagineering and constructed by Intamin.
</item 4>

<item 5>
Lasseter Family Winery
Lasseter Family Winery is a winery located in Glen Ellen, Sonoma County, California.

 The winery was founded in 2000 by Pixar and Walt Disney Animation Studios CCO John Lasseter and his wife, Nancy Lasseter.

 The winery, once inhabited by the Grand Cru Winery, produces approximately 1,200 cases of French red wine blends annually, with the capacity to produce up to
6,000.

 The winery grows Bordeaux and Rhône varietals on 27 acres.

 One of the Lasseters' winemaking mentors was Jess Jackson, of Kendall-Jackson.
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
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 7>

<item 8>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
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
Red's Dream
Red's Dream is a 1987 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The short film, which runs four minutes, stars Red, a unicycle.

 Propped up in the corner of a bicycle store on a rainy night, Red dreams about a better place.

 "Red's Dream" was Pixar's second computer-animated short following "Luxo Jr." in 1986, also directed by Lasseter.
</item 10>
=== END ITEM id=5ec43833e3202226 ===

=== ITEM id=f37f0bbfb52d212f ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Nicolas Cage starred in a variety of films.
  Fact 0.1: Nicolas Cage starred in Valley Girl (1983).
  Fact 0.2: Nicolas Cage starred in Racing with the Moon (1984).
  Fact 0.3: Nicolas Cage starred in Birdy (1984).
  Fact 0.4: Nicolas Cage starred in Peggy Sue Got Married (1986).
  Fact 0.5: Nicolas Cage starred in Raising Arizona (1987).
  Fact 0.6: Nicolas Cage starred in Moonstruck (1987).
  Fact 0.7: Nicolas Cage starred in Vampire's Kiss (1989).
  Fact 0.8: Nicolas Cage starred in Wild at Heart (1990).
  Fact 0.9: Nicolas Cage starred in Fire Birds (1990).
  Fact 0.10: Nicolas Cage starred in Honeymoon in Vegas (1992).
  Fact 0.11: Nicolas Cage starred in Red Rock West (1993).
Sentence 1:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 1.0: The film is about a widowed 37-year-old Italian-American woman.
  Fact 1.1: Cher plays the widowed 37-year-old Italian-American woman.
  Fact 1.2: The widowed woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 1.3: The widowed woman's fiancé is played by Danny Aiello.
  Fact 1.4: The fiancé's younger brother is estranged.
  Fact 1.5: The fiancé's younger brother is hot-tempered.
  Fact 1.6: The fiancé's younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 1>

<item 2>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 2>

<item 3>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 3>

<item 4>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 4>

<item 5>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 5>

<item 6>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 6>

<item 7>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 7>

<item 8>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 8>

<item 9>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 9>

<item 10>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 10>
=== END ITEM id=f37f0bbfb52d212f ===

=== ITEM id=045fd9fa8b0a75c7 ===
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
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 2>

<item 3>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 3>

<item 4>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 4>

<item 5>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 5>

<item 6>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 6>

<item 7>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 7>

<item 8>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 8>

<item 9>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 9>

<item 10>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 10>
=== END ITEM id=045fd9fa8b0a75c7 ===

=== ITEM id=ab65a99de49777e7 ===
Gold sentences and facts:
Sentence 0: Dieffenbachia is a genus of tropical flowering plants in the family Araceae.
  Fact 0.0: Dieffenbachia is a genus of flowering plants.
  Fact 0.1: Dieffenbachia is a genus of tropical flowering plants.
  Fact 0.2: Dieffenbachia is in the family Araceae.
Sentence 1: Carlina is a genus of flowering plants in the aster family, Asteraceae.
  Fact 1.0: Carlina is a genus of flowering plants.
  Fact 1.1: Carlina is in the aster family.
  Fact 1.2: Carlina is in the family Asteraceae.

Retrieved items:
<item 1>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 1>

<item 2>
Carlina
Carlina is a genus of flowering plants in the aster family, Asteraceae.

 It is distributed from Madeira and the Canary Islands across Europe and northern Africa to Siberia and northwestern China.
</item 2>

<item 3>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 3>

<item 4>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 4>

<item 5>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 5>

<item 6>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 6>

<item 7>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 7>

<item 8>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 8>

<item 9>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 9>

<item 10>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 10>
=== END ITEM id=ab65a99de49777e7 ===

=== ITEM id=65cc45867b2a73fe ===
Gold sentences and facts:
Sentence 0: Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.
  Fact 0.0: Remember the Titans is a 2000 American sports film
  Fact 0.1: Remember the Titans was produced by Jerry Bruckheimer
  Fact 0.2: Remember the Titans was directed by Boaz Yakin
Sentence 1: My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and Ray Walston, based on the 1960s television series of the same name in which Walston starred.
  Fact 1.0: My Favorite Martian is a 1999 American science fiction comedy film
  Fact 1.1: My Favorite Martian stars Christopher Lloyd
  Fact 1.2: My Favorite Martian stars Jeff Daniels
  Fact 1.3: My Favorite Martian stars Daryl Hannah
  Fact 1.4: My Favorite Martian stars Elizabeth Hurley
  Fact 1.5: My Favorite Martian stars Wallace Shawn
  Fact 1.6: My Favorite Martian stars Ray Walston
  Fact 1.7: My Favorite Martian is based on the 1960s television series of the same name
  Fact 1.8: Ray Walston starred in the 1960s television series My Favorite Martian
Sentence 2:  The film grossed $36.8 million domestically against a budget of $65 million.
  Fact 2.0: The film (My Favorite Martian) grossed $36.8 million domestically
  Fact 2.1: The film (My Favorite Martian) had a budget of $65 million

Retrieved items:
<item 1>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 1>

<item 2>
My Favorite Martian (film)
My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and
Ray Walston, based on the 1960s television series of the same name in which Walston starred.

 It was directed by Donald Petrie and written by Sherri Stoner and Deanna Oliver, based on the television series created by John L. Greene.

 Creatures were created by Amalgamated Dynamics from designs by Jordu Schell.

 The film grossed $36.8 million domestically against a budget of $65 million.
</item 2>

<item 3>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 3>

<item 4>
Managed Money
Managed Money is a 1934 short comedy film directed by Charles Lamont.

 The film stars Frank Coghlan Jr. and Shirley Temple.

 It was also known as "Frolics of Youth" and "Measured Money".

 This was the second film in which Temple starred as Mary Lou.

 The film tells the story of Sonny and Sid prospecting for gold to pay for their military academy education.

 Sonny's sister, Mary Lou stows away in the back of their car.

 Eventually, they meet an inventor who helps them financially.
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
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 6>

<item 7>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 7>

<item 8>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 8>

<item 9>
Lay the Favorite
Lay the Favorite (promoted as Lay the Favourite in the UK) is a 2012 American comedy-drama film starring Bruce Willis, Rebecca Hall, Catherine Zeta-Jones and
Joshua Jackson.

 Based on Beth Raymer's memoir of the same title, the film follows a young, free-spirited woman as she journeys through the legal and illegal world of sports
gambling.

 The film was directed by Stephen Frears.
</item 9>

<item 10>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 10>
=== END ITEM id=65cc45867b2a73fe ===

=== ITEM id=51af38ce87673390 ===
Gold sentences and facts:
Sentence 0:  He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.
  Fact 0.0: He won three singles titles.
  Fact 0.1: He won 26 doubles titles.
  Fact 0.2: He won the Australian Open 1978 doubles title with Wojtek Fibak.
  Fact 0.3: He won the Australian Open 1980 title with fellow countryman Mark Edmondson.
  Fact 0.4: He won the Australian Open 1981 title with fellow countryman Mark Edmondson.
  Fact 0.5: He won the Roland Garros 1986 title with fellow countryman Mark Edmondson.
  Fact 0.6: He was a runner-up at the Australian Open 1985 with fellow countryman Mark Edmondson.
  Fact 0.7: Mark Edmondson is a fellow countryman of his.
Sentence 1:  In his career, he won nine doubles titles, but no singles titles.
  Fact 1.0: In his career, he won nine doubles titles.
  Fact 1.1: In his career, he won no singles titles.

Retrieved items:
<item 1>
Evonne Goolagong career statistics
This is a list of the main career statistics of Australian former tennis player Evonne Goolagong.

 During her career, which lasted from 1967 to 1983, Goolagong won seven singles titles at a Grand Slam event and was a runner-up on 11 occasions.

 In addition she won five Grand Slam doubles titles, partnering Margaret Court, Peggy Michel and Helen Gourlay, as well as one mixed doubles title with Kim
Warwick.

 In total she won 82 singles titles, 46 doubles titles and 4 mixed doubles titles.

 She achieved a No. 1 singles ranking for a two-week period in April–May 1976 although this was only officially recognized in 2007.

 She was a member of the Australian Federation Cup teams that won the cup in 1971, 1973 and 1974.
</item 1>

<item 2>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 2>

<item 3>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 3>

<item 4>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 4>

<item 5>
Rossana de los Ríos
Rossana de los Ríos (born 16 September 1975) is a retired professional tennis player from Paraguay.

 She was born in the Paraguayan capital, Asunción, and now lives in Miami, USA, with her daughter Ana Paula Neffa de los Ríos and her husband Gustavo Neffa, a
retired Paraguayan Association football player who played for Boca Juniors.

 Rossana achieved a career high singles ranking of No. 51 on 12 September 2001, and No. 52 in doubles on the Sony Ericsson WTA Tour.

 De los Ríos played doubles with Grand Slam champions such as Maria Sharapova, Svetlana Kuznetsova, Arantxa Sánchez Vicario, as well as Jelena Janković and
Dinara Safina.

 Rossana is also a former world number one Junior player, having won the 1992 Roland Garros Juniors event.

 De los Ríos participated in six editions (1994, 2001, 2002, 2003, 2009 and 2010) of the Australian Open.
</item 5>

<item 6>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 6>

<item 7>
Samantha Stosur career statistics
This is a list of the main career statistics of professional Australian tennis player, Samantha Stosur.

 To date, Stosur has won eight WTA singles titles including one Grand Slam singles title at the 2011 US Open.

 She was also the runner-up at the 2010 French Open and a semi-finalist at the 2010 and 2011 WTA Tour Championships.

 Stosur has also enjoyed a successful doubles career, in which she has held the World No. 1 ranking and won twenty-four WTA doubles titles including two grand
slam women's doubles titles at the 2005 US Open and 2006 French Open and two year-ending championships at the 2005 and 2006 WTA Tour Championships.
</item 7>

<item 8>
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
</item 8>

<item 9>
1990 Stockholm Open – Doubles
Jorge Lozano and Todd Witsken were the defending champions, but lost in the quarterfinals this year.
</item 9>

<item 10>
1989 Volvo International – Doubles
Jorge Lozano and Todd Witsken were the defending champions but did not compete that year.
</item 10>
=== END ITEM id=51af38ce87673390 ===

=== ITEM id=01ca6781bcd0300a ===
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
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 2>

<item 3>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 3>

<item 4>
Deadfall (2012 film)
Deadfall is a 2012 American crime drama film directed by Stefan Ruzowitzky, written by Zach Dean and starring Eric Bana, Olivia Wilde, and Charlie Hunnam.
</item 4>

<item 5>
Winnie the Pooh (2011 film)
Winnie the Pooh is a 2011 American animated buddy musical comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 51st Disney animated feature film.

 Inspired by A. A. Milne's stories of the same name, the film is part of Disney's "Winnie the Pooh" franchise, the fifth theatrical "Winnie the Pooh" film
released, and Walt Disney Animation Studios' second adaptation of "Winnie-the-Pooh" stories.

 Jim Cummings reprises his vocal roles as Winnie the Pooh and Tigger, while series newcomers Travis Oates, Tom Kenny, Craig Ferguson, Bud Luckey, and Kristen
Anderson-Lopez provide the voices of Piglet, Rabbit, Owl, Eeyore, and Kanga, respectively.

 In the film, the aforementioned residents of the Hundred Acre Wood embark on a quest to save Christopher Robin from an imaginary culprit while Pooh deals with
a hunger for honey.
</item 5>

<item 6>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 6>

<item 7>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 7>

<item 8>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 8>

<item 9>
Toy Story 3
The film features an ensemble voice cast with Tom Hanks, Tim Allen, Joan Cusack, Don Rickles, Wallace Shawn, John Ratzenberger, Estelle Harris, Jodi Benson, and
John Morris reprising their roles from the previous films, along with Blake Clark as the voice of Slinky Dog (because Jim Varney died ten years before the
release of the film) and Ned Beatty, Michael Keaton, Whoopi Goldberg, Timothy Dalton, Kristen Schaal, Bonnie Hunt, and Jeff Garlin voicing the new characters
introduced in this film.
</item 9>

<item 10>
Are We Done Yet?
Are We Done Yet?

 is a 2007 American family comedy film starring Ice Cube.

 The film is a remake of the Cary Grant comedy "Mr. Blandings Builds His Dream House" and a sequel to the 2005 comedy "Are We There Yet?

" The film was directed by Steve Carr from a screenplay by Hank Nelken.

 It was produced by Revolution Studios and RKO Pictures and distributed by Columbia Pictures.

 The film was shot on location in Tsawwassen, British Columbia, Canada, but is set in Newberg, Oregon, United States.
</item 10>
=== END ITEM id=01ca6781bcd0300a ===

=== ITEM id=5e4cd19c3b40f8a3 ===
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
Nitish Roy
Nitish Roy is an Indian film art director, production designer, and costume designer in Hindi cinema and a Bengali Film Director, who is also known for his work
with art cinema directors, Shyam Benegal, Mrinal Sen and Govind Nihlani, Hindi mainstream cinema, directors like Rajkumar Santoshi, as well as international
directors like Mira Nair and Gurinder Chadha.
</item 1>

<item 2>
The Cat in the Hat (film)
Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.

 It is based on the 1957 Dr. Seuss book "of the same name".

 The film stars Mike Myers in the title role of the Cat in the Hat, and Dakota Fanning as Sally.

 Sally's brother (who is unnamed in the book and the 1971 TV special), Conrad, is portrayed by Spencer Breslin.

 The film is the second feature-length Dr. Seuss adaptation after the 2000 holiday film "How the Grinch Stole Christmas".
</item 2>

<item 3>
Graph query row:
creative_work: Dr. Seuss' The Cat in the Hat
relationship: CREATED_BY
person: Bo Welch
occupation: director
</item 3>

<item 4>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 4>

<item 5>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 5>

<item 6>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 6>

<item 7>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 7>

<item 8>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
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
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 10>
=== END ITEM id=5e4cd19c3b40f8a3 ===

=== ITEM id=d120440ab313bb20 ===
Gold sentences and facts:
Sentence 0: Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.
  Fact 0.0: Frederick King Keller was born in 1954.
  Fact 0.1: Frederick King Keller was born in Buffalo, New York.
  Fact 0.2: Frederick King Keller is an American director.
  Fact 0.3: Frederick King Keller is an American producer.
  Fact 0.4: Frederick King Keller is an American screenwriter.
  Fact 0.5: Frederick King Keller works in film and television.
Sentence 1: Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
  Fact 1.0: Ian Schneider was born on October 3, 1992.
  Fact 1.1: Ian Schneider is an independent American film director.
  Fact 1.2: Ian Schneider is an American screenwriter.

Retrieved items:
<item 1>
Graph query row:
person: Frederick King Keller
occupation: ['director', 'producer', 'screenwriter']
relationship: None
career_role: None
related_entity: None
</item 1>

<item 2>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 2>

<item 3>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 3>

<item 4>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 4>

<item 5>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 5>

<item 6>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 6>

<item 7>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 7>

<item 8>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 8>

<item 9>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 9>

<item 10>
Graph query row:
person: Ian Schneider
occupation: ['independent film director', 'screenwriter']
relationship: None
career_role: None
related_entity: None
</item 10>
=== END ITEM id=d120440ab313bb20 ===

=== ITEM id=89ef05ffc9f1e9e9 ===
Gold sentences and facts:
Sentence 0: Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister, and for two terms President of the General Assembly of the League of Nations (1930–32).
  Fact 0.0: Nicolae Titulescu was born on March 4, 1882.
  Fact 0.1: Nicolae Titulescu died on March 17, 1941.
  Fact 0.2: Nicolae Titulescu was a well-known Romanian diplomat.
  Fact 0.3: Nicolae Titulescu was at various times a government minister.
  Fact 0.4: Nicolae Titulescu was at various times finance minister.
  Fact 0.5: Nicolae Titulescu was at various times foreign minister.
  Fact 0.6: Nicolae Titulescu served for two terms as President of the General Assembly of the League of Nations.
  Fact 0.7: Nicolae Titulescu's two terms as President of the General Assembly of the League of Nations are dated 1930–32.
Sentence 1: The League of Nations (abbreviated as LN in English, "La Société des Nations " ] abbreviated as SDN or SdN in French) was an intergovernmental organisation founded on 10 January 1920 as a result of the Paris Peace Conference that ended the First World War.
  Fact 1.0: The League of Nations is abbreviated as LN in English.
  Fact 1.1: The League of Nations is abbreviated as SDN or SdN in French.
  Fact 1.2: The French name of the League of Nations is 'La Société des Nations'.
  Fact 1.3: The League of Nations was an intergovernmental organisation.
  Fact 1.4: The League of Nations was founded on 10 January 1920.
  Fact 1.5: The League of Nations was founded as a result of the Paris Peace Conference.
  Fact 1.6: The Paris Peace Conference ended the First World War.

Retrieved items:
<item 1>
Ghimpețeni
Ghimpețeni is a commune in Olt County, Romania.

 It is composed of two villages, Ghimpețeni and Ghimpețenii Noi.

 These were part of Nicolae Titulescu Commune until 2004, when they were split off.
</item 1>

<item 2>
Remus Nicolai
Remus Nicolai (b. June 10, 1977 in Bistriţa, Romania) is a retired Romanian aerobic gymnast.

 He had a successful career winning six world championships medals (one gold, two silver and three bronze) and five European championships medals (two gold, two
silver and one bronze).

 After retiring from aerobic gymnastics he opened together with his wife (Daniela Mărănducă) a private gymnastics club in Constanţa.

 As of 2013 he trains, together with his wife, the junior artistic women's National Team of Romania at the National Olympic Center in Onesti.
</item 2>

<item 3>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 3>

<item 4>
Nicolae Titulescu
Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister,
and for two terms President of the General Assembly of the League of Nations (1930–32).
</item 4>

<item 5>
Ioan Silviu Suciu
Ioan Silviu Suciu (born 24 November 1977 in Sibiu) is a retired Romanian artistic gymnast.

 His best event was the pommel horse.

 One of his closest rivals was his team mate and pommel horse Olympic champion Marius Urzică, who was defeated by Suciu in Ljubljana, 2004, when the latter won
the title while Urzică took 8th place.

 Suciu is an Olympic bronze medalist with the team, a silver world medalist on pommel horse and a six-time European medalist (pommel horse, vault and team).

 Suciu was one of the key team members of the Romanian gymnastics team for several years contributing to the 2004 Olympic team bronze medal and three
continental team medals (two gold and one silver).

 He also placed fourth all around at the 2004 Olympic Games.
</item 5>

<item 6>
Marius Urzică
Marius Daniel Urzică (born September 30, 1975, in Topliţa, Romania) is a Romanian gymnast.

 Urzică is an Olympic champion, a three-time world champion and a three-time European champion on pommel horse.

 He competed at three Olympic games, medaling each time on pommel horse (gold Sydney 2000, silver Atlanta 1996 and silver Athena 2004) and contributed to the
team bronze in Athens 2004.

 His unique technique and style of performance have won him the recognition as one of the greatest masters on this piece of apparatus ever, together with
Miroslav Cerar and Zoltán Magyar.

 Known as "The King of the Pommels" in 2001 he achieved the maximum score of 10.00 on this piece of apparatus at the Glasgow Grand Prix.

 Two elements in artistic gymnastics, one on pommel horse and one on parallel bars are named after him.
</item 6>

<item 7>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 7>

<item 8>
Graph query row:
organization: League of Nations
date_founded: 10 January 1920
</item 8>

<item 9>
Gheorghe Tătărescu
During his first time in office, he moved closer to King Carol II, leading an ambivalent policy toward the fascist Iron Guard and ultimately becoming
instrumental in establishing the authoritarian and corporatist regime around the National Renaissance Front.

 In 1940, he accepted the cession of Bessarabia and Northern Bukovina to the Soviet Union, and consequently had to resign.
</item 9>

<item 10>
Gheorghe Tătărescu
Gheorghe I. Tătărescu (also known as "Guță Tătărescu", with a slightly antiquated pet form of his given name; 2 November 1886 – 28 March 1957) was a Romanian
politician who served 36th Prime Minister of Romania (1934–1937; 1939–1940), three times as Minister of Foreign Affairs ("interim" in 1934 and 1938; appointed
to the office in 1945-1947), and once as Minister of War (1934).

 Representing the "young liberals" faction inside the National Liberal Party (PNL), Tătărescu began his political career as a collaborator of Ion G. Duca,
becoming noted for his anti-Communism and, in time, for his conflicts with the PNL's leader Dinu Brătianu and the Foreign Minister Nicolae Titulescu.
</item 10>
=== END ITEM id=89ef05ffc9f1e9e9 ===
