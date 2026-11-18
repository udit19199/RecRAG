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

=== ITEM id=1838466b7b581a61 ===
Gold sentences and facts:
Sentence 0: Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.
  Fact 0.0: Like many of Marivaux's plays, La Surprise de l'amour makes use of stock characters from the Commedia dell'arte.
  Fact 0.1: Many of Marivaux's plays make use of stock characters from the Commedia dell'arte.
Sentence 1:  In this play, Arlequin and Columbine are featured.
  Fact 1.0: In La Surprise de l'amour, Arlequin is featured.
  Fact 1.1: In La Surprise de l'amour, Columbine is featured.
Sentence 2: Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the Italian "Commedia dell'arte".
  Fact 2.0: Harlequin is the best-known of the zanni, the comic servant characters from the Italian Commedia dell'arte.
  Fact 2.1: Harlequin is called Arlecchino in Italian.
  Fact 2.2: Harlequin is called Arlequin in French.
  Fact 2.3: Harlequin is called Harlequin in Old French.

Retrieved items:
<item 1>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 1>

<item 2>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 2>

<item 3>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 3>

<item 4>
Gilles (stock character)
Gilles (] )—sometimes Gille—is a stock character of French farce and Commedia dell'Arte.

 He enjoyed his greatest vogue in 18th-century France, in entertainments both at the fairgrounds of the capital and in private and public theaters, though his
origins can be traced back to the 17th century and, possibly, the century previous.

 A "zanni", or comic servant, he is a type of bungling clown, stupid, credulous, and lewd—a character that shares little, problematically, with the sensitive
figure in Watteau's famous portrait that, until the latter half of the 20th century, bore his name alone.

 Gilles fades from view in the 19th century, to persist in the 20th and 21st as the Belgian Gilles of Binche Carnival.
</item 4>

<item 5>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 5>

<item 6>
L'Arlésienne (Bizet)
Georges Bizet composed L'Arlésienne as incidental music to Alphonse Daudet's play of the same name, usually translated as "The Girl from Arles".

 It was first performed on 1 October 1872 at the Vaudeville Theatre (now a cinema known as the Gaumont Opéra).

 Bizet's music consists of 27 numbers (some only a few bars) for voice, chorus, and small orchestra, ranging from short solos to longer entr'actes. Bizet
himself played the harmonium backstage at the premiere performance.
</item 6>

<item 7>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 7>

<item 8>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 8>

<item 9>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
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
=== END ITEM id=1838466b7b581a61 ===

=== ITEM id=e16b4df7edcccacf ===
Gold sentences and facts:
Sentence 0: Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France (1933–1940 and 1950–1957), and the United States (1947–1950).
  Fact 0.0: Maximillian Oppenheimer was born on 6 May 1902.
  Fact 0.1: Maximillian Oppenheimer died on 26 March 1957.
  Fact 0.2: Maximillian Oppenheimer was known as Max Ophüls.
  Fact 0.3: Max Ophüls was a German-born film director.
  Fact 0.4: Max Ophüls worked in Germany from 1931 to 1933.
  Fact 0.5: Max Ophüls worked in France from 1933 to 1940.
  Fact 0.6: Max Ophüls worked in France from 1950 to 1957.
  Fact 0.7: Max Ophüls worked in the United States from 1947 to 1950.
Sentence 1: Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.
  Fact 1.0: Shirley Clarke was born on October 2, 1919.
  Fact 1.1: Shirley Clarke died on September 23, 1997.
  Fact 1.2: Shirley Clarke was an American experimental filmmaker.
  Fact 1.3: Shirley Clarke was an American independent filmmaker.

Retrieved items:
<item 1>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 1>

<item 2>
Graph query row:
person: Max Ophüls
death_date: 26 March 1957
</item 2>

<item 3>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 3>

<item 4>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 4>

<item 5>
Charlie O'Connell (roller derby)
Charlie O'Connell (May 7, 1935 – February 9, 2015) was a New York City roller derby skater, considered the premier male star of his sport.

 He was inducted into the Roller Derby Hall of Fame in 1967, after his first retirement.
</item 5>

<item 6>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 8>

<item 9>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 9>

<item 10>
Graph query row:
person: Shirley Clarke
death_date: September 23, 1997
</item 10>
=== END ITEM id=e16b4df7edcccacf ===

=== ITEM id=daa77cbd3f8af41a ===
Gold sentences and facts:
Sentence 0:  Her only novel, "The Bone People", won the Booker Prize in 1985.
  Fact 0.0: Her only novel is 'The Bone People'.
  Fact 0.1: 'The Bone People' won the Booker Prize.
  Fact 0.2: 'The Bone People' won the Booker Prize in 1985.
Sentence 1:  His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science fiction's top U.S. prizes in consecutive years.
  Fact 1.0: Card's novel 'Ender's Game' was published in 1985.
  Fact 1.1: 'Speaker for the Dead' is the sequel to 'Ender's Game'.
  Fact 1.2: 'Speaker for the Dead' was published in 1986.
  Fact 1.3: 'Ender's Game' won the Hugo Award.
  Fact 1.4: 'Ender's Game' won the Nebula Award.
  Fact 1.5: 'Speaker for the Dead' won the Hugo Award.
  Fact 1.6: 'Speaker for the Dead' won the Nebula Award.
  Fact 1.7: Card is the only author to win both science fiction's top U.S. prizes in consecutive years.

Retrieved items:
<item 1>
List of Ender's Game series short stories
There are various sources for short stories set in the Ender's Game series.

 One is the short story collection "First Meetings" by Orson Scott Card.

 This collection contains the original novelette "Ender's Game" plus three other stories.

 Another source is Card’s webzine "InterGalactic Medicine Show".

 The first four stories from Card's webzine: "Mazer in Prison," "Pretty Boy," "Cheater," and "A Young Man with Prospects," also appear in the paperback
anthology "Orson Scott Card's InterGalactic Medicine Show".

 Reprints of short stories in the Ender's Game series can be found in other science fiction anthologies.
</item 1>

<item 2>
Ender in Exile
"Ender in Exile" begins one year after Ender has won the bugger war, and begins with the short story "Ender's Homecoming" from Card's webzine Intergalactic
Medicine Show.

 Other short stories that were published elsewhere are included as chapters of the novel.
</item 2>

<item 3>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 3>

<item 4>
Ender Wiggin
Andrew "Ender" Wiggin is a fictional character from Orson Scott Card's 1985 science fiction novel "Ender's Game" and its sequels ("Speaker for the Dead",
"Xenocide", "Children of the Mind", "Ender in Exile"), as well as in the first part of the spin-off series, "Ender's Shadow".

 The book series itself is an expansion, with some changes to detail, of Card's 1977 short story "Ender's Game."
</item 4>

<item 5>
Shadow of the Hegemon
Shadow of the Hegemon (2001) is the second novel in the "Ender's Shadow" series (often called the Bean Quartet) by Orson Scott Card.

 It is also the sixth novel in the "Ender's Game" series.

 It is told mostly from the point of view of Bean, a largely peripheral character in the original novel "Ender's Game" but the central protagonist of the
parallel narrative "Ender's Shadow."

 "Shadow of the Hegemon" was nominated for a Locus Award in 2002.
</item 5>

<item 6>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 6>

<item 7>
A War of Gifts: An Ender Story
A War of Gifts: An Ender Story (2007) is a science fiction novel by Orson Scott Card.

 This book is set in Card's Ender's Game series and takes place during Ender Wiggin's time at Battle School as described in Card's novels "Ender's Game" and
"Ender's Shadow".
</item 7>

<item 8>
Ender's Game (short story)
"Ender's Game" is a story by Orson Scott Card.

 It first appeared in the August 1977 issue of "Analog" magazine and was later expanded into the novel "Ender's Game".

 Although the foundation of the Ender's Game series, the short story is not properly part of the "Ender's Game" universe, as there are many discrepancies in
continuity.
</item 8>

<item 9>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 9>

<item 10>
Ender in Exile
Ender in Exile is a science fiction novel by Orson Scott Card, part of the "Ender's Game" series, published on November 11, 2008.

 It takes place between the two award-winning novels: "Ender's Game" and "Speaker for the Dead".

 It could also be considered a parallel novel to the first three sequels in the Shadow Saga, since the entirety of this "trilogy" takes place in the span of
"Ender in Exile".

 The novel concludes a dangling story line of the Shadow Saga, while it makes several references to events that take place during the Shadow Saga.

 From yet another perspective, the novel expands (or "replaces") the last chapter of the original novel "Ender's Game".

 On the one hand, it fills the gap right before the last chapter, and on the other hand, it fills the gap between the last chapter and the original (first)
sequel (both named "Speaker for the Dead").
</item 10>
=== END ITEM id=daa77cbd3f8af41a ===

=== ITEM id=79dac61376856c56 ===
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
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
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
=== END ITEM id=79dac61376856c56 ===

=== ITEM id=5a6766a0875bf85c ===
Gold sentences and facts:
Sentence 0:  He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.
  Fact 0.0: This person creates bespoke millinery under the J Smith Esquire brand.
  Fact 0.1: This person creates bespoke millinery for a portfolio of private clients.
  Fact 0.2: The portfolio of private clients includes Angelina Jolie.
  Fact 0.3: The portfolio of private clients includes Brad Pitt.
  Fact 0.4: The portfolio of private clients includes Amal Clooney.
Sentence 1: Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and human rights.
  Fact 1.0: Amal Clooney is also known by the name Alamuddin (née Alamuddin).
  Fact 1.1: Amal Clooney's Arabic name is أمل علم الدين.
  Fact 1.2: Amal Clooney was born on 3 February 1978.
  Fact 1.3: Amal Clooney is a barrister at Doughty Street Chambers.
  Fact 1.4: Amal Clooney specialises in international law.
  Fact 1.5: Amal Clooney specialises in human rights.

Retrieved items:
<item 1>
Doughty Street Chambers
Doughty Street Chambers is a British set of barristers' chambers situated in Bristol, Manchester and London's Doughty Street, undertaking criminal justice,
public law, immigration, employment, human rights and civil liberties work.
</item 1>

<item 2>
Amal Clooney
Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and
human rights.

 Her clients include Julian Assange, the founder of WikiLeaks, in his fight against extradition.

 She has also represented the former prime minister of Ukraine, Yulia Tymoshenko, and Egyptian-Canadian journalist Mohamed Fahmy.

 She is married to the American actor George Clooney.
</item 2>

<item 3>
Pupil master
A pupil master ('pupilmaster' or 'pupil-master') or in the case of a female barrister, 'pupil mistress' etc., is the former name given to an experienced
barrister who a pupil shadows during their pupillage.

 The term pupil master or mistress has now been replaced by the term 'pupil supervisor'.

 Barristers are called to the Bar via one of the four Inns of Court upon successful completion of the BPTC and having undertaken a required number of
"qualifying sessions" in their chosen Inn of Court.

 In most cases, the newly called barrister is then required to undertake training for a period of at least a year before the barrister can start their own
private practice.

 This training period is known as pupillage, usually split into two periods of six months known as "sixes".

 The first "six" is a non-practising six, during which the pupil will shadow their pupil master; the second is usually a practising "six", when the pupil, with
their pupil master's permission, can undertake the supply of legal services and exercise rights of audience in court.
</item 3>

<item 4>
Pupil master
Occasionally, a pupil barrister may undertake a third "six", extending the training period a further six months.

 At the end of pupillage, to continue practising the law, a barrister may attempt to become a tenant in a set of barristers' chambers, or find a position as an
employed barrister.
</item 4>

<item 5>
Mónica Feria Tinta
Mónica Feria Tinta is a leading public international lawyer.

 She practises as a Barrister at 20 Essex Street Chambers.

 In 2000 she became the first Peruvian-born lawyer to receive the Diploma of the Hague Academy of International Law in history.

 Her litigation work led to the first international human rights court decision ordering the prosecution of a former Head of State for crimes under
international law.

 She was awarded the Inge Genefke International Award for her work as an international lawyer in 2006 and was co-recipient of the Gruber Justice Prize 2007, for
her contributions advancing the cause of justice as delivered through the legal system.

 She is a member of the Bar of England and Wales, and the American Society of International Law.
</item 5>

<item 6>
Shahram Taghavi
Shahram Taghavi is a barrister practising in the United Kingdom.

 He is a specialist in Judicial Review, Human Rights, Immigration and EU law.

 He was called to the England & Wales bar in 1994.

 He is a partner at Article 1, a law firm specialising in immigration, EU freedoms of movement and human rights law.

 He practised as an independent barrister at Doughty Street Chambers specialising in immigration and public law.

 He was head of both the Immigration and Public Law departments at Simons Muirhead & Burton solicitors.

 He was a Senior Barrister at Bates Wells & Braithwaite LLP as a member of their Public & Regulatory and Immigration departments, Deputy Head of Immigration at
Lewis Silkin LLP, Head of Immigration at Charles Russell LLP (now Charles Russell Speechly LLP) and National Head of Immigration at Grant Thornton LLP.
</item 6>

<item 7>
Guglielmo Verdirame
Guglielmo Verdirame (born in Reggio di Calabria, Italy) is a Professor of International Law at King's College London in the Department of War Studies and the
School of Law.

 He was previously a university lecturer in law at the University of Cambridge, Faculty of Law, and a Fellow of the Lauterpacht Centre for International Law.

 He practises as a barrister at 20 Essex Street Chambers.
</item 7>

<item 8>
Barbados Bar Association
The Barbados Bar Association is a voluntary association of attorneys in Barbados who practise at the independent bar as barristers and Queen's Counsel.

 It was created by the Barbados Bar Association Act of 1940.
</item 8>

<item 9>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 9>

<item 10>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 10>
=== END ITEM id=5a6766a0875bf85c ===

=== ITEM id=677453cb21fa2ffa ===
Gold sentences and facts:
Sentence 0: Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
  Fact 0.0: Marc Rosset was born on 7 November 1970.
  Fact 0.1: Marc Rosset is a former professional tennis player.
  Fact 0.2: Marc Rosset is from Switzerland.
  Fact 0.3: Marc Rosset is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
Sentence 1: Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional tennis player.
  Fact 1.0: Maksim Mirnyi, also known as Max Mirnyi, was born on 6 July 1977 in Minsk.
  Fact 1.1: Maksim Mirnyi's Belarusian name is Максім Мікалаевіч Мірны.
  Fact 1.2: Maksim Mirnyi's Russian name is Максим Николаевич Мирный.
  Fact 1.3: Maksim Mirnyi is a Belarusian professional tennis player.

Retrieved items:
<item 1>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 1>

<item 2>
Max Mirnyi
Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional
tennis player.
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
Vitas Gerulaitis
Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.

 In 1975, Gerulaitis won the men's doubles title at Wimbledon, partnering with Sandy Mayer.

 He won the men's singles title at one of the two Australian Open tournaments held in 1977 (Gerulaitis won the tournament that was held in December, while
Roscoe Tanner won the earlier January tournament).

 Gerulaitis also won two Italian Open titles, in 1977 and 1979, and the WCT Finals in Dallas, in 1978.
</item 4>

<item 5>
2010 Australian Open – Mixed Doubles
Sania Mirza and Mahesh Bhupathi were the defending champions.

 They accepted the invitation for the tournament, but withdrew before their match against Carly Gullickson and Bernard Tomic and were replaced by Akgul
Amanmuradova and Rik de Voest.
</item 5>

<item 6>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 6>

<item 7>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 7>

<item 8>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 8>

<item 9>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 9>

<item 10>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 10>
=== END ITEM id=677453cb21fa2ffa ===

=== ITEM id=27a393318920abe2 ===
Gold sentences and facts:
Sentence 0:  At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
  Fact 0.0: She won a silver medal at the 2002 Commonwealth Games.
  Fact 0.1: She won the silver medal with Sara Petersen in the women's doubles at the 2002 Commonwealth Games.
  Fact 0.2: She won a bronze medal in the mixed team at the 2002 Commonwealth Games.
Sentence 1: The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25 July to 4 August 2002.
  Fact 1.0: The 2002 Commonwealth Games were officially known as the XVII Commonwealth Games.
  Fact 1.1: The 2002 Commonwealth Games were commonly known as Manchester 2002.
  Fact 1.2: The 2002 Commonwealth Games were held in Manchester, England.
  Fact 1.3: The 2002 Commonwealth Games were held from 25 July to 4 August 2002.

Retrieved items:
<item 1>
1982 U.S. Pro Indoor – Doubles
Marty Riessen and Sherwood Stewart were the defending champions, but Riessen did not participate this year.

 Stewart partnered Ferdi Taygan, finishing runner-up.
</item 1>

<item 2>
Nicolas N’Godrela won three tennis golds at the 2015 Pacific Games. Aerobic gymnast Remus Nicolai earned 11 world and European medals, then coached in Romania.
Brett McClure won Olympic team silver and is USA Gymnastics’ men’s High Performance Director. Masayoshi Yamamoto won 2014 Asian Games pommel horse gold. Louis
Smith won five Olympic medals across three Games and a 2015 European pommel horse title. Luke
</item 2>

<item 3>
1983 Donnay Indoor Championships – Doubles
Pavel Složil and Sherwood Stewart were the defending champions, but Složil did not participate this year.

 Stewart partnered Ferdi Taygan, losing in the first round.
</item 3>

<item 4>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 4>

<item 5>
Nicole Gordon (badminton)
Nicole Gordon (born 17 March 1976 in Lower Hutt, New Zealand) is a female badminton player from New Zealand.

 At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
</item 5>

<item 6>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 6>

<item 7>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 7>

<item 8>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 8>

<item 9>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 9>

<item 10>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 10>
=== END ITEM id=27a393318920abe2 ===

=== ITEM id=f63fc1948cae380d ===
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
The Short-Timers
The Short-Timers is a 1979 semi-autobiographical novel by U.S. Marine Corps veteran Gustav Hasford, about his experience in the Vietnam War.

 It was later adapted into the film "Full Metal Jacket" (1987) by Hasford, Michael Herr, and Stanley Kubrick.

 Hasford's novel "The Phantom Blooper" (1990) is a sequel to "The Short-Timers".

 The book was supposed to be the first of a trilogy, but Hasford died soon after completing its sequel and before writing the third installment.
</item 1>

<item 2>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 2>

<item 3>
Time Bomb High School
Time Bomb High School is the second album by the Reigning Sound.

 It was released in 2002 on In the Red Records.

 The album featured the original Reigning Sound lineup of Greg Cartwright on lead vocals and guitar; Alex Greene on organ, piano, guitar, and backing vocals;
Jeremy Scott on bass, and backing vocals; and Greg Roberson on drums.

  Howlin' Pelle Almqvist of The Hives commented in "Rolling Stone" that Time Bomb High School was his favorite record of 2002.
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
Still Kicking: The Fabulous Palm Springs Follies
Still Kicking: The Fabulous Palm Springs Follies is a 1997 American short documentary film directed by Mel Damski.

 It was nominated at the 70th Academy Awards for Best Documentary Short Subject.

 It features The Fabulous Palm Springs Follies, which was formerly staged at the Palm Springs, California Plaza Theatre.
</item 5>

<item 6>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 6>

<item 7>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
</item 7>

<item 8>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 8>

<item 9>
The Time to Live and the Time to Die
The Time to Live and the Time to Die, also known as A Time to Live, A Time to Die is a 1985 film directed by Hou Hsiao-hsien.

 This film is inspired by screenwriter-turned-director Hou's own coming-of-age story.
</item 9>

<item 10>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 10>
=== END ITEM id=f63fc1948cae380d ===

=== ITEM id=3f7ff60934bded6e ===
Gold sentences and facts:
Sentence 0: The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.
  Fact 0.0: The Battle of Saipan was a battle of the Pacific campaign of World War II.
  Fact 0.1: The Battle of Saipan was fought on the island of Saipan in the Mariana Islands.
  Fact 0.2: The Battle of Saipan was fought from 15 June to 9 July 1944.
Sentence 1:  The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
  Fact 1.0: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division were commanded by Lieutenant General Holland Smith.
  Fact 1.1: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division defeated the 43rd Infantry Division of the Imperial Japanese Army.
  Fact 1.2: The 43rd Infantry Division of the Imperial Japanese Army was commanded by Lieutenant General Yoshitsugu Saito.
Sentence 2:  It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.
  Fact 2.0: The battle referred to here was fought by American and Filipino forces against Japanese troops.
  Fact 2.1: The battle referred to here took place in Manila, the capital city of the Philippines.

Retrieved items:
<item 1>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 1>

<item 2>
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
</item 2>

<item 3>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 3>

<item 4>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 4>

<item 5>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 5>

<item 6>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 6>

<item 7>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 7>

<item 8>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 8>

<item 9>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 9>

<item 10>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 10>
=== END ITEM id=3f7ff60934bded6e ===

=== ITEM id=4d51fb4eabbf8f78 ===
Gold sentences and facts:
Sentence 0:  The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.
  Fact 0.0: The album includes a collaboration with Popcaan.
  Fact 0.1: The album includes a collaboration with Zhu.
  Fact 0.2: The album includes a collaboration with Leikeli47.
  Fact 0.3: The album includes a collaboration with Dreezy.
  Fact 0.4: The album includes a collaboration with Flume.
  Fact 0.5: The album includes a collaboration with Pell.
Sentence 1: Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of 2014, signed to Mind of a Genius Records and Columbia Records.
  Fact 1.0: Steven Zhu was born in 1989.
  Fact 1.1: Steven Zhu is known professionally as Zhu.
  Fact 1.2: Zhu is a Chinese American electronic musician.
  Fact 1.3: Zhu is a Chinese American singer.
  Fact 1.4: Zhu has been active since the beginning of 2014.
  Fact 1.5: Zhu is signed to Mind of a Genius Records.
  Fact 1.6: Zhu is signed to Columbia Records.

Retrieved items:
<item 1>
Shela moved from band FBI to Avex solo work, Sunny-side Up, and a 2009 comeback. The Breeders’ “Cannonball” reached #44 in the US and #2 on alternative charts.
Mazzy Star formed from Opal, with Hope Sandoval replacing Kendra Smith. Vektroid’s 2011 Macintosh Plus album *Floral Shoppe* popularized vaporwave. Cherushii
was an influential San Francisco producer who died in 2016
</item 1>

<item 2>
Kid Confucius
Kid Confucius were an eight-piece Australian band from Sydney, Australia.

 They formed in 2001 and have since played well over 400 live shows around the country, including major festivals as well as their own headline shows at venues
such as The Annandale Hotel and The Metro Theatre.

 In 2005 the band released a self-titled album and two singles "Words" and "Skintight."

 The album, a mish-mash of soul, hip-hop, pop and funk, received great critical acclaim, most notably from Rolling Stone who hailed the album as one of the
standout local releases of the year.

 "Words" also enjoyed some solid months of radio and TV play.

 The band released its second album Stripes in 2007 with three singles "Closer", "Last Straw" and "Moment".

 Stripes was the band's attempt at making a Detroit-era Motown soul album and it was quick to earn rave reviews from press around the country as well as a
publishing deal with Mushroom.

 "Moment" has enjoyed good radio play on triple j. Kid Confucius is set to release its third album in October 2008.

 The first single from this album, "Good Luck", is out now.
</item 2>

<item 3>
BT’s Grammy-nominated *These Hopeful Machines* (2010) was a two-disc electronic album with guest vocalists; it spawned shortened and remix editions.
AlunaGeorge’s *I Remember* (2016) featured several collaborators, and its Popcaan-assisted single “I’m in Control” reached UK No. 39. Berlin’s *Pleasure Victim*
(1982/83), marking Terri Nunn’s return, became their best-selling, platinum-certified
</item 3>

<item 4>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 4>

<item 5>
Method Man
Clifford Smith (born April 1, 1971), better known by his stage name Method Man, is an American rapper, record producer, and actor.

 He is known as a member of the East Coast hip hop collective Wu-Tang Clan.

 He is also one half of the hip hop duo Method Man & Redman.

 He took his stage name from the 1979 film "Method Man".

 In 1996, he won a Grammy Award for Best Rap Performance by a Duo or Group, for "I'll Be There for You/You're All I Need to Get By", with American R&B
singer-songwriter Mary J. Blige.
</item 5>

<item 6>
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 6>

<item 7>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 7>

<item 8>
Candy Chen
Candy Chen (陳斯亞 born January 3, 1993) is a dancer, actress, host, singer, rapper and a model.

 She is featured on the Taiwanese variety shows: Blackie's Teenage Club and Blackie Lollipop.

 Candy was a member of the Taiwanese girl group Hey Girl from 2010 to 2011, and a member of from 2013 to 2016.

 She is 1/8th Dutch.

 She changed her name to Chen Yi-Ling in July 2012 to follow her mother's maiden name.

 Her English name becomes Nina Chen.

 In November 2013, she changed her name to Candy Chen (English Name) and Chen Si Ya (Chinese Name).
</item 8>

<item 9>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 9>

<item 10>
I Remember (AlunaGeorge album)
I Remember is the second studio album by English electronic music duo AlunaGeorge.

 It was released on 16 September 2016 by Island Records.

 The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.

 The album's lead single, "I'm in Control", was released in February 2016, featuring vocals from rapper Popcaan, and peaked at number 39 on the UK Singles
Chart.
</item 10>
=== END ITEM id=4d51fb4eabbf8f78 ===

=== ITEM id=01d6af3e04508feb ===
Gold sentences and facts:
Sentence 0:  His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed according to its experimental approach.
  Fact 0.0: His experimental works are well known in the Malayalam Cinema industry.
  Fact 0.1: His experimental works are appropriately discussed in the Malayalam Cinema industry.
  Fact 0.2: His debut movie is Jalachhayam.
  Fact 0.3: Jalachhayam was released in 2010.
  Fact 0.4: Jalachhayam was well discussed according to its experimental approach.
Sentence 1: Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA, Britain and Germany.
  Fact 1.0: Jacques Feyder was born on 21 July 1885.
  Fact 1.1: Jacques Feyder died on 24 May 1948.
  Fact 1.2: Jacques Feyder was a Belgian actor.
  Fact 1.3: Jacques Feyder was a Belgian screenwriter.
  Fact 1.4: Jacques Feyder was a Belgian film director.
  Fact 1.5: Jacques Feyder worked principally in France.
  Fact 1.6: Jacques Feyder also worked in the USA.
  Fact 1.7: Jacques Feyder also worked in Britain.
  Fact 1.8: Jacques Feyder also worked in Germany.

Retrieved items:
<item 1>
Sathish Kalathil
Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam.

 He is also Story Writer, and Lyricist.

 His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed
according to its experimental approach.

 Actually, his first venture was in 2008 named Veena Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in movie.

 In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about Municipal solid waste.
</item 1>

<item 2>
Jalachhayam
Jalachhayam (മലയാളം: ) is a 2010 Indian Malayalam-language Experimental film, produced and directed by Sathish Kalathil under the banner of 'The People's
Films'.

 The story was written by Sujith Aalungal and describes the warm relationship between a village man and an artist (painter) from the city.
</item 2>

<item 3>
Shajoon Kariyal
Shajoon Kariyal is an Indian film director and producer working in Malayalam cinema.

 Shajoon was born in 1963 in Kozhikode, Kerala and had his primary education from Govt.

 Ganapath High School, Chalappuram.

 He started his film career in 1984, at the age of 18, as an assistant director to I. V. Sasi.

 He worked as the assistant or associate director to many films including "Uyarangalil" (1984), "Anubandham" (1985), "Karimpinpoovinakkare" (1985), "Aavanazhi"
(1986), "1921" (1988), "Douthyam" (1989), "Varthamana Kalam" (1990), "Arhatha" (1990), "Midhya" (1991), "Neelagiri" (1991) and "Varnapakittu" (1997).

 He was the story writer for the Mammootty-starrer megahit "Jackpot" (1993).
</item 3>

<item 4>
Jacques Feyder
Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA,
Britain and Germany.

 He was a leading director of silent films during the 1920s, and in the 1930s he became associated with the style of poetic realism in French cinema.

 He adopted French nationality in 1928.
</item 4>

<item 5>
Laloorinu Parayanullathu
Laloorinu Parayanullathu (What Has Laloor To Say) (മലയാളം: ) is an Indian documentary film directed by Sathish Kalathil in Malayalam Language.

 The documentary narrates about the problems of Municipal Garbage dumping in Laloor, a suburban area in Thrissur Municipal Corporation.

 The film reveals the history of the waste dumping in Laloor since the period of Sakthan Thampuran who was the ruler of the Kingdom of Cochin and first violence
against the Garbage dumping here.
</item 5>

<item 6>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 6>

<item 7>
Shamal Sabri
Shamal Sabri (Kurdish: شه‌مال سه‌بری), (born 9 November 1985), is an independent Kurdish award-winning film producer and Duhok International Film Festival
Artistic Director, graduated from Duhok University with BA in English Literature.

.

 He was born in the city of Duhok (Kurdish: دهوك) in Iraqi Kurdistan.

 Apart from being one of the co-founders of Duhok IFF in 2009, he has worked in many other fields of video productions, produced many commercials and music
videos in NV Production and Vin TV.

 Throughout his film carrier, he has contributed in many award-winning films as an assistant director, production manager, assistant producer and line producer.

 His latest contributions were in many films such as "Before Snowfall" by Hisham Zaman, "The Swallow" by Mano Khalil, "Memories on Stone" by Shawkat Amin Korki
and Michael by Kurdo Duski.
</item 7>

<item 8>
T. Muthuraj
T. MuthuRaj (born July 29, 1969, Coimbatore, India) is an Indian film production designer.

 He entered the film industry after completing his B.Sc.

 degree in fine arts as first assistant to Sabu Cyril after which he has served as production designer in critically acclaimed and celebrated films in major
Indian languages.

 His big break came in the 1997 Malayalam film "Guru" starring Mohanlal and Suresh Gopi, directed by Rajiv Anchal, for which he received the Kerala State Film
Award for Best Art Director.

 Muthuraj has since worked in such path breaking hits as "Arputha Theevu", "Angaditheru", "Pazhasi Raja", "Irumbukottai Murattu Singham" and "Avan Ivan".

 He has contributed to the success of 43 feature films and 185 commercials.

 He has won many film awards for best art direction and for his creative works.

 His latest work is "Nanban" (2012) directed by Shankar, which has won huge appreciation from all quarters.
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
Shajoon Kariyal
He debuted as a director with "Rajaputhran" (1996), starring Suresh Gopi, Shobhana and Vikram.

 He has directed many films, including the commercially successful "Thachiledathu Chundan" (1999) and the critically acclaimed "Vadakkumnadhan" (2006).

 After "Vadakkumnadhan", he planned two films, "Raman Police" and "Talkies", but both the projects did not work out.

 In 2012, he directed "Chettayees" which he also co-produced, as one of the five partners of the newly launched production house Thakkaali Films.

 His latest film is "Sir C. P." (2015).
</item 10>
=== END ITEM id=01d6af3e04508feb ===

=== ITEM id=a80a93dc256f29db ===
Gold sentences and facts:
Sentence 0: The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the headquarters of book publisher Random House and a luxury apartment complex.
  Fact 0.0: The Random House Tower is also known as the Park Imperial Apartments.
  Fact 0.1: The Random House Tower is a 52-story mixed-use tower.
  Fact 0.2: The Random House Tower is in New York City, United States.
  Fact 0.3: The Random House Tower is used as the headquarters of book publisher Random House.
  Fact 0.4: The Random House Tower is used as a luxury apartment complex.
Sentence 1:  It currently carries the Vornado Realty Trust corporate headquarters.
  Fact 1.0: The Random House Tower currently carries the Vornado Realty Trust corporate headquarters.

Retrieved items:
<item 1>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 1>

<item 2>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 2>

<item 3>
Flagami
The Flagami is a neighborhood in Miami, Florida, United States, roughly defined as south and east of the Tamiami Canal, north of the Tamiami Trail (US 41/South
Eighth Street), and west of Red Road (SR 959/West 57th Avenue), bisected by Flagler Street.
</item 3>

<item 4>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 4>

<item 5>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 5>

<item 6>
Villard (imprint)
Villard, also known as Villard Books, is a publishing imprint of Random House, one of the largest publishing companies in the world.

 It was founded in 1983.

 Villard began as an independent imprint of Random House and is currently a sub-imprint of Ballantine Books, itself an imprint of Random House.

 It was named after a Stanford White brownstone mansion on Madison Avenue that was the home of Random House for twenty years.
</item 6>

<item 7>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 7>

<item 8>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 8>

<item 9>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 9>

<item 10>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 10>
=== END ITEM id=a80a93dc256f29db ===
