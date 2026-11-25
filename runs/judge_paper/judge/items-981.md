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

This file has 2 items.

=== ITEM id=59ee0b2dedd2e7db ===
Gold sentences and facts:
Sentence 0: This isn’t the first time that Jada has opened up about her long-term commitment to her husband.
  Fact 0.0: This is not the first time Jada has opened up about her long-term commitment to her husband.
Sentence 1: Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus.
  Fact 1.0: Jada praised her children for how they helped her grow.
  Fact 1.1: Jada said: 'My children, they're little gurus.'
Sentence 2: She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep sense of self-acceptance”.
  Fact 2.0: Jada and Will have two children.
  Fact 2.1: Jada and Will's children are Jaden, who is 25, and Willow, who is 22.
  Fact 2.2: Jada praised her children for being her 'little gurus' throughout this time.
  Fact 2.3: Jada's children taught her 'a deep sense of self-acceptance'.

Retrieved items:
<item 1>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
However, there are still certain behaviours in a marriage that could signal it’s time to get a divorce.

“If you feel disrespected by your partner, feel humiliated, or were cheated on, you probably feel ready to split,” Lucas said. “I also think a real big one is,
if you lose sexual attraction for your partner and you’ve actively tried to get it back and can’t, it’s time to go.”
</item 1>

<item 2>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“It’s hard... the thing Will and I talk about a lot is the journey. We started in this [relationship] at a very young age, 22 years old... I think you expect
your partner to know - especially when it comes to sex. It’s like: ‘Well, if you love me, you should know. If you love me, you should read my mind.’ That’s a
huge pitfall,” she said at the time.

The next month, Will’s memoir is released which details several issues the couple have had over the years. He wrote they were “suffering the brutal death of our
romantic fantasies, the burning away of the idealistic illusion of the perfect marriage and the perfect family. Neither of us wanted a divorce; we knew we loved
each other, and some aspects of our union were magical. But the structure of the life that we had established was strangling both of us.”
</item 2>

<item 3>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Will and Jada with their son Jayden in 2003 (Getty)

2013: Jada denies open marriage rumours (sort of)

After a quiet decade and a half making movies and raising their children, Jada took to Facebook to address rumours she and Will were in an open relationship.

She wrote: “Here is how I will change my statement... Will and I BOTH can do WHATEVER we want, because we TRUST each other to do so. This does NOT mean we have
an open relationship... this means we have a GROWN one.”

2015: Will addresses divorce rumours

Will addresses persistent separation rumours by clarifying the pair were not getting a divorce.

“Under normal circumstances, I don’t usually respond to foolishness. But, so many people have extended [to] me their ‘deepest condolences’ that I figured -
‘What the hell… I can be foolish, too!’” he wrote on Facebook at the time.
</item 3>

<item 4>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
They welcomed two sons together, Preston and Jayden, before divorcing in 2007. At the time of the divorce, they split custody, but a few months later, Federline
would receive sole physical custody because of what a judge called Spears' "habitual, frequent and continuous use of controlled substances and alcohol."

In 2008, when Spears began experiencing trouble with her mental health, Federline was awarded sole legal custody as well. Spears was given visitation rights. In
her recently released memoir, "The Woman in Me," she claimed that she agreed to her conservatorship for her kids.

"Because I played by the rules, I was reunited with my boys," Spears explained.

She eventually gained more custody, though Federline has remained the kids' primary caregiver. They have lived with him since the divorce. Earlier this year,
they moved to Hawaii with Spears' approval.

After years of getting along, some tension began to show in 2022.
</item 4>

<item 5>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

A bride-to-be has been defended after choosing her mother to walk her down the aisle instead of her stepfather.

In a recent post shared to the popular “Am I The A**hole?” Reddit forum, the mother – who goes by the username u/PopShort7064 – asked if she was in the wrong
for agreeing to walk her child down the aisle, in addition to doing a mother/daughter dance at the wedding.
</item 5>

<item 6>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
I want to know who they’re with, where they’re going. Even now that they’re driving, I want them to text me when they arrive and when they’re heading home."

LIKE WHAT YOU’RE READING? CLICK HERE FOR MORE ENTERTAINMENT NEWS

Bure explained that they "work hard to keep our communication open and talk through any situation" and they are "always willing to listen."

"But I don’t always say yes," she added. "My goal as a parent is not to make my kids happy but to guide them to be respectful, curious, responsible, resilient,
kind, giving and thoughtful adults."

NATASHA BURE PRAISES MOM CANDACE CAMERON FOR HER FAITH, BEING 'BOLD' AMID BACKLASH

Kristen Bell and Dax Shepard

Kristen Bell and Dax Shepard share two daughters together, Lincoln, 10, and Delta, 8, and have been married for 10 years.
</item 6>

<item 7>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She suffers from postnatal depression and feels hopeless about her ability to protect her sons: “I got a little depressed once I was no longer keeping them safe
inside my body … I wanted them back inside me so the world couldn’t get at them.”

Every part of normal life had been stripped from me. I had no freedom and yet also no security

She has a keen sense of injustice regarding a perceived contract that she never signed with the public and the press. “They just kept acting like I owed it to
them to let the men who kept trying to catch me looking fat take photos of my infant sons.” Her postpartum body and shots of her without makeup were treated as
“some kind of a sin – as if gaining weight was something unkind I’d done to them personally, a betrayal. At what point did I promise to stay 17 for the rest of
my life?”
</item 7>

<item 8>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
he wrote on Facebook at the time. “So, in the interest of redundant, repetitious, over and over-again-ness… Jada and I are… NOT GETTING A DIVORCE!!!!!!!!!!!!!”

2018: Will revealed that he and Jada don’t refer to themselves as married

In an episode of TIDAL’s Rap Radar podcast, Will said he and Jada “refer to ourselves as life partners, where you get into that space where you realise you are
literally with somebody for the rest of your life. There’s no deal breakers. There’s nothing she could do - ever. Nothing that would break our relationship. She
has my support till death and it feels so good to get to that space.”
</item 8>

<item 9>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
“Listen, weddings are beautiful, but they can be very romanticised,” Jada said. “I feel that was a very real moment for the two of us to look each other in the
eyes, recognise that there would be tough times in this journey and to say to each other: ‘No matter what, we’re going to figure it out and that’s why we don’t
need a prenup, because I’m making a promise that divorce won’t be necessary, that we will figure this out.’”

The Girls Trip star explained how she and Will casually promised not to get a divorce, and how that decision impacted their perspective on on marriage.

“And we made that promise to each other without all of the bridal wedding beauty; it was just sitting on a log in his mother’s backyard, and going: ‘Hey’ -
having to really look at the possibility of us not being together,” she said.

This isn’t the first time that Jada has opened up about her long-term commitment to her husband.
</item 9>

<item 10>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
She specified that she also has a husband, who’s been a stepfather to her daughter since she was 10.

The parent went on to give more context about her family, noting that she lost the father of her children - Calla and Hawthorn - when they were six and eight.
She then shared that she remarried four years later, before opening up about her relationship with her second husband and how her children reacted to it.

“We were friends for several months, dated some, stronger feelings developed and I introduced him to the kids to see how they would get along,” she wrote. “We
halted for a year while my kids and I did some therapy because their reaction to my husband was strongly negative, because they didn’t want to replace their
dad. But once therapy was ongoing, they were doing better. From there, things move faster but the kids were on board for things to move on.”
</item 10>

<item 11>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
Michelle Andrews wears Friends With Frank “Marlowe” blazer, $549, and “Sabine” trousers, $399. Veja shoes, $240, from Hype DC. Credit: Hugh Stewart In many
ways, the relationship between Shameless Media and its audience is much like that between the friends. It began with a shared interest in celebrity gossip and
TV shows, before moving into more personal recommendations of books and podcasts, until eventually reaching the stage where life’s most personal questions can
be asked and answered. Ask Shameless, which lands in more than 64,000 subscriber inboxes each Friday, answers reader questions about work, dating, friendship
and more. The best of these advice columns have been published in a new book, 4am: Answers to the conundrums that keep you up at night. Andrews and McDonald’s
second book feels markedly different to their first, The Space Between.
</item 11>

<item 12>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
He went on to explain that he’s “happier than [he’s] ever been in [his] entire life” and praised Jada for continuing to be by his side.

“I have achieved every single thing I have ever wanted in my entire life. There is not one thing left that I desire in this world - all of the awards, all of
the money, the family - everything I’ve ever dreamed,” he said. “And those dreams were largely built on the foundation of Jada’s sacrifices. There were many
times when Jada put her career aside so I could follow the dreams of mine.”

The Fresh Prince of Bel-Air star thanked his wife for putting him “before herself” and for maintaining a good relationship with his ex-wife Sheree Zampino. Will
and Zampino share a 30-year-old son, Trey Smith, who was also present for the book event at the Enoch Pratt Free Library.
</item 12>

<item 13>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
He later issued an apology to Spears on Instagram, writing: “I’ve seen the messages, tags, comments, and concerns and I want to respond. I am deeply sorry for
the times in my life where my actions contributed to the problem, where I spoke out of turn, or did not speak up for what was right. I understand that I fell
short in these moments and in many others and benefited from a system that condones misogyny and racism.”

In June 2021, he also shared his support for Spears after she testified publicly in a bid to end her conservatorship, which was controlled by her father, Jamie
Spears. In a Los Angeles court hearing at the time, Spears alleged that she had been forced to take lithium against her will. She also claimed that the team
managing her conservatorship had prevented her from having another baby by refusing to remove her IUD.

“After what we saw today, we should all be supporting Britney at this time.
</item 13>

<item 14>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
MILA KUNIS AND ASHTON KUTCHER LAUGH OFF CRITICISM RECEIVED FOR THEIR BATHING ROUTINE CONFESSION

Candace Cameron Bure

"Full House" star Candace Cameron Bure is a proud mom of three with husband Valeri Bure: Natasha, Lev and Maksim, who are now all in their early 20s.

When her kids were teenagers, Bure spoke with L.A. Parent about her parenting style, noting that she can be "strict."

"I hope my kids would say that I’m a loving mom," Bure said. "They would also describe me as strict. We set firm boundaries, but there is always love and
grace."

She continued, "I’m the mom that needs to meet the parents before my kids can go to a new friend’s house. I want to know who they’re with, where they’re going.
</item 14>

<item 15>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
To help keep their kids grounded, Kunis revealed in a 2020 interview on "The Late Show with Stephen Colbert" how she keeps their privilege in check.

"I like to remind them that Mommy and Daddy have money, but they’re broke all the time," she said with a laugh. "That is not their bedroom, it is my bedroom
that I’m letting them use, so they better take good care of it."

She continued, "I say this in jest, but really they don’t have things … they’re raised on hand-me-downs from girlfriends’ kids’ clothing; we garden."

Kunis noted her kids are lucky to be born into a famous family and wants to ensure they are grateful.

"Our kids are super-fortunate," she said. "Regardless, they were born at the right time to the right people, and I love them, but they’re not going to be
entitled a--holes."
</item 15>

<item 16>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 16>

<item 17>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Timberlake, she likens herself to Benjamin Button. “Somehow that year, in becoming more vulnerable I started to feel like a child again.”

The situation worsens amid postnatal depression. “It was as if some part of me became the baby,” she writes in an apologetic section about transferring her
frustrations on to her interior decorator. “One part of me was a very demanding grown woman yelling about white marble, while another part of me was suddenly
very childlike.”

When she is placed under the conservatorship, she not only regresses, but feels dehumanised by the constant scrutiny. “I became a robot. But not just a robot –
a sort of child-robot. The conservatorship stripped me of my womanhood, made me into a child.”
</item 17>

<item 18>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
and Sarah Michelle Gellar

"Buffy the Vampire Slayer" star Sarah Michelle Gellar and husband Freddie Prinze Jr. share two kids together and have been married for more than 20 years.

In a 2021 interview with Us Weekly, Prinze gave some details on how they parent their kids, now ages 13 and 11.

"We are strict as hell," the "I Know What You Did Last Summer" star told the outlet. "Look, I don’t believe in whooping kids. I’ve never whooped mine. My mom
whooped me. They were, like, preventative whooping, so I know how to be strict."

Prinze added that Gellar also knows how to be strict and has "100 rules" for their kids.
</item 18>

<item 19>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.
</item 19>

<item 20>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
You just get this profound feeling of instability." He stopped drinking altogether in 2013.

Micheál and Daniel, now 28 and 27, are both actors. Micheál took on his mother's surname of Richardson, something Neeson said was "the right thing" to do.

"I think it was a lovely homage, a nice gesture and he's, you know, he's not saddled with my last name which, I have a certain celebrity status I guess," Neeson
told Andy Cohen in 2020. "I'd hate for him to be constantly asked, 'Oh, are you Liam Neeson's son?' So, it was a lovely gesture. Natasha's family, mother and
sisters, were very touched by it, as indeed I was, too."

Kevin Federline

Kevin Federline met Britney Spears in 2004 at a nightclub, and months later they were married.
</item 20>

<item 21>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
If Tasha was here, someone could share this.' But yeah, we’re doing all right, you know?"

He said, "The thing Tasha and I really wanted to instill in them was manners. That may sound like such a basic thing, but I’ve heard so many adults say, 'Oh, my
God, your sons — they look you straight in the eye. They’re very polite and mannered.' It makes me proud. And it can get you through a lot, in a way. Just
showing respect for your fellow man."

LIAM NEESON'S ‘RETRIBUTION’ ACTION HERO ROLE LATEST IN STORIED CAREER MARKED BY LOVE AND LOSS

Neeson also admitted to "drinking too much" after losing his wife. In another 2014 interview, this one with "60 Minutes," he said that grief sometimes hit him
"like a wave. You just get this profound feeling of instability."
</item 21>

<item 22>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
The couple has been honest about the trials and tribulations of raising their kids, especially to keep them grounded.

"It’s very important to me that they share a bedroom. I think their lives will be easier than most other people’s on the planet, and to develop a good
character, it’s important to always be going through something," Bell told Self in 2020.

"I like the fact that they will have to figure out how to share a bedroom, figure out how to share your closet, figure out how to share your space," she
continued. "If that’s the worst thing about your life, that you have to share a bedroom with your sister, you’re going to be OK."

CLICK HERE TO GET THE FOX NEWS APP

Bell also laid out her strict screen-time rules, but she admits they get a little lax on the weekends.

"They’re not allowed to play with our phones or have any screens really.
</item 22>

<item 23>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
(AFP via Getty Images)

“They’re not resolving their problems, they’re rescheduling them. So, if you separate from someone, but you live in a separate house and you’re still together,
it seems to me you’re just not letting go of the past because you want to stick to what’s familiar,” he claimed. “People stick to what they know, and it’s like
you have a tether to that person. Your brain’s like: ‘I cannot let go of them because I am scared of the unknown.’ I don’t think I’ve ever seen a couple
separate and come back again.”

It’s unclear whether Jada and Will plan on ending their separation and living in the same home again. However, the couple have been candid about their plans to
not get a divorce. Jada previously recalled that when she first decided to get married to Will in 1997, it would be for the long haul.
</item 23>

<item 24>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
While it was wonderful to have the emotional and practical support, after three months I felt that If I didn’t get out there to live as an independent person, I
would quickly lose any sense of what I was capable of. But I’m incredibly grateful for the time I had living with my parents; I know it’s a huge privilege to
have them as a safety net. I ended up writing a memoir, Inconceivable, about my journey to become a solo mum – and the complexities of being a woman in your
late 30s who has to start over and move back in with your parents. I think while we’re all embracing this contemporary feminist movement where women can create
families of their own making, we’re also realising that we need help, that parenthood can’t happen alone.”
</item 24>

<item 25>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”
</item 25>

<item 26>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
But the structure of the life that we had established was strangling both of us.”

2022: No infidelity statement and Oscars incident

During an interview on CBS Sunday Morning earlier this month, Will said he and Jada had never accused each other of cheating.

He said: “Never. There’s never been infidelity in our marriage...And we have never surprised one another with anything ever..I have decided that chatter about
my life can be of a benefit to people. I think that chatter is the first stage to having a real conversation and being able to truly explore if some of the
things in your heart are loving, or poisonous.”

In March 2022, Will slapped Chris Rock at the 94th annual Oscars before he yelled at the comedian to “keep my wife’s name out your f***ing mouth”. It quickly
became the most talked-about moment of the evening with the pair initially appearing to laugh off the tense moment.

The altercation came after Rock said: “Jada, I love you.
</item 26>

<item 27>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
That same day, the fitness instructor officially filed for divorce, listing the date of separation as 28 July 2023 and citing "irreconcilable differences" as
the reason for the split.

In a statement posted online, Asghari said: “After six years of love and commitment to each other my wife and I have decided to end our journey together. We
will hold onto the love and respect we have for each other and I wish her the best always. S*** happens. Asking for privacy seems ridiculous so I will just ask
for everyone including the media to be kind and thoughtful.”

Days before Spears’ memoir release date, Asghari revealed that he had obtained an early copy of the book. In an interview with TMZ on 13 October, he explained
that he wasn’t worried about anything being written in the book that could paint him in a negative light.

“I already read it and I’m very proud of her,” he told the outlet.
</item 27>

<item 28>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
And I went through the differences about my longevity and her limited life."

JOHN TRAVOLTA HONORS LATE WIFE KELLY PRESTON ON MOTHER'S DAY WITH SWEET THROWBACK VIDEO

He continued, "I said, 'Ben, you have always loved the truth, and I'm going to tell you the truth about life. Nobody knows when they are going to go or when
they are going to stay. Your brother left at 16. Too young. Your mother left at 57. That was too young. But who's to say? I could die tomorrow. You could.
Anybody can. So let's look at it like it's part of life. You don't know exactly. You just do your best at trying to live the longest you can."
</item 28>

<item 29>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Although they’ve made their dedication to each other quite clear, there have been some grey areas over the last seven years about the terms of Will and Jada’s
separation. In 2020, Jada made headlines when she revealed in a Red Table Talk episode with her husband that she had a relationship with Alsina. She explained
that she and her husband were “separated amicably” when the romance occurred, which had not been disclosed to the press at the time. “We decided that we were
going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,” Jada said. As the Girls
Trip star recalled that she “got into a different kind of entanglement with August,” she was prompted by Will to confirm it “was a relationship, absolutely.”
</item 29>

<item 30>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
"They’re not allowed to play with our phones or have any screens really. They are allowed to watch TV on the weekends, and we’re pretty liberal with it, which
I’ve been considering paying more attention to," she said. "But since they’re not allowed at all during the week, they will wake up at 6:30 on a Saturday
morning, and they will watch TV until noon. I will also sleep until 9:45, which is really nice, and then get stuff done."

The "Frozen" star added, "But I will say, I notice how hard it is to pull them away from the TV at noon."
</item 30>

<item 31>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Perhaps one of the reasons why Jada and Will have promised not to get a divorce is because they also agreed not to sign a prenuptial agreement. In an interview
with Parade published on 13 October, Jada explained it was her special connection with Will that encouraged them not to sign a prenup before getting married.

“I feel that was a very real moment for the two of us to look each other in the eyes, recognise that there would be tough times in this journey and to say to
each other: ‘No matter what, we’re going to figure it out and that’s why we don’t need a prenup, because I’m making a promise that divorce won’t be necessary,
that we will figure this out,’” she recalled. “And we made that promise to each other without all of the bridal wedding beauty; it was just sitting on a log in
his mother’s backyard, and going: ‘Hey’ - having to really look at the possibility of us not being together.”
</item 31>

<item 32>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
He said turning to Jesus for a deeper meaning and learning to not take his "first thoughts" for fact was crucial to his overall mental health.

"The Word of God is living and active, bringing strength, courage, wisdom, and protection," he said.

"God's thoughts and ways are higher than ours, but when we spend time, each day reading His Word and listening to the voice of the Good Shepherd, we are filled
with new thoughts that change the trajectory of our lives."

Bradley also shared his dos and don'ts of positive thinking.

He said it's not "denial and pretending that there are no challenges," he said.

"Positive thinking is not flattery and frivolous compliments."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Rather, positive thinking is realistic, full of truth and love — and aligned with heaven and scripture.

"Positive thinking is not self-consumption, self-absorption or selfishness. Positive thinking is not creating unrealistic expectations for other people and
putting extra stress on them to be perfect."
</item 32>

<item 33>
Want to help your kids achieve their potential? This expert has some advice (The Age, 2023-12-01T18:00:00+00:00)
Loading How do you know if you’ve achieved your potential? We often look to extrinsic rewards – titles, money, awards – as the metrics of success. But, Grant
argues: “They are status symbols that we use to broadcast to others and to ourselves that we matter and I think they are really poor proxies for mattering. I
think we have all been in situations where we can admire people not for what they’ve achieved but for who they’ve become or what they’ve contributed to others.”
</item 33>

<item 34>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
Read this article for free! Plus get unlimited access to thousands of articles, videos and more with your free account! Please enter a valid email address. By
entering your email, you are agreeing to Fox News Terms of Service and Privacy Policy , which includes our Notice of Financial Incentive . To access the
content, check your email and follow the instructions provided.

Charlie Sheen is speaking out about his current experience with fatherhood.

The actor claims that he is currently raising his two youngest children, 14-year-old twins Max and Bob, on his own. As he explained to People, most of his time
lately has been spent on "doing single dad stuff."

The boys' mother, Brooke Mueller, has struggled with drug addiction for several years. He said that recently she "has been trying to figure some stuff out on
her end, so she's not in the picture too much right now."
</item 34>

<item 35>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
Converse shoes, $130, from Hype DC. Credit: Hugh Stewart Says McDonald, “If I was to psychoanalyse us, I’d say we felt guilty for having that platform that
young.” But that imposter syndrome is nowhere to be seen in 4am, in which Andrews and McDonald are happy to take their rightful place as agony aunts to young
Australian women. In it, readers trust Andrews, McDonald and other Shameless columnists with their quandaries on everything from infidelity to sibling rivalry.
The book comes at a time when many young women have trust issues. TikTok is serving eating-disorder content to those already struggling, influencers and
celebrities post undisclosed advertisements, and algorithms work to push everything from egg-freezing to injectables. Everywhere young women look, someone is
making money off them. Zara McDonald (left) wears Venroy mini dress, $200. Michelle Andrews (right) wears Venroy shirt, $280, and pants, $280.
</item 35>

<item 36>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.
</item 36>

<item 37>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
They’re really exploring something vastly different. It doesn’t end up that you go back into that little box of ‘this is a marriage, we’re living together’. It
just doesn’t do that.”

Every relationship is different, living in the public eye or not. Although living separately may not be an issue for some couples, there can still be some major
signs that it’s time to leave a relationship behind. “When there is no stone left unturned, or when they have tried every single combination possible to make
the relationship function and yet it doesn’t, that’s the time to pull the plug,” Winter added. “They’ve done their individual work and they have done the work
together and ended up in the same place.”

Will and Jada’s relationship could end in a variety of ways. The celebrity couple could stay separated, move back in together as a married pair, or they can get
a divorce.
</item 37>

<item 38>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
This isn’t the first time that Jada and Will have taken some time apart from each other. During a 2020 episode of her Facebook Watch series, Red Table Talk,
Jada disclosed that while she and Will were separated amicably, she had a relationship with singer August Alsina, describing their involvement as an
“entanglement”.

Throughout their marriage of 26 years, the A-listers have made it clear that divorce was never in the cards. Jada furthered that belief during a preview for an
interview with Today’s Hoda Kotb, which aired on 13 October. “I made a promise that there will never be a reason for us to get a divorce,” she said. “We will
work through, whatever. I just haven’t been able to break that promise.”

Of course, outsiders aren’t fully aware of the “heavy-duty work” that Jada and Will have done throughout their marriage.
</item 38>

<item 39>
Emily Ratajkowski’s latest photo shoot sparks fury (The Independent - Life and Style, 2023-11-30T20:02:31+00:00)
Famous plus-size models and inclusivity advocates - such as Tess Holliday, Kelly Augustine, Stephanie Yeboah, Letticia Munniz, and Sarah Chiwaya - have called
out the photo for emphasising Ratajkowski’s thinness in a pair of pants that would typically be reserved for someone who is plus-size.

“I’ve been looking for those jeans in the second photo if you could just please return them that would cool. Tysm,” Holliday commented under Ratajkowski’s post.

“That second photo is such godforsaken trash,” wrote Chiwaya. “In the year of our lord 2023?? STILL??? Yikes.”

“What in the fatphobic hell is that second picture?” a third user commented, while someone else said: “That second pic is so out of touch and ignorant but the
comments are chef’s kiss.”

Plus-size models have long advocated for more inclusive sizing - both in stores and on the runway.
</item 39>

<item 40>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
In an interview withWSJ Magazine published on 20 November, Kelce described how much he admires Swift’s skills as a musician. “I’ve never been a man of words.
Being around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

Although he acknowledged the public attention his personal life has received because he’s “never dated anyone with that kind of aura around them,” Kelce
specified that he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life.

“When she acts like that I better not be the one acting all strange.”
</item 40>

<item 41>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
CHARLIE SHEEN'S ‘SURREAL’ TV RETURN COMES AFTER HE COMPARED DRUG, ALCOHOL ADDICTION TO ‘DEMONIC POSESSION’

Mueller's rep declined to comment on the situation to Fox News Digital.

Sheen, who also has three daughters — one with a high school sweetheart and two with ex-wife Denise Richards — called his sons "really cool, really smart and
really funny," but admitted they do have a bit of a problem "worshiping their devices."

He laughed, "Although on the flip side, when they're doing that it gives me a little time to not be dragging them all over the city. But I also feel like the
screens detract them from having other opportunities to expand their minds in other areas."

Sheen went through his own period of addiction several years ago — Richards famously took the twins in at one point when Sheen and Mueller were both deemed
incapable of caring for them.
</item 41>

<item 42>
Author Melissa Lucashenko on playing with black and white binaries (The Age, 2023-10-06T04:50:00+00:00)
The family came under the extra scrutiny of authorities after Copson shot and injured a man who tried to rape her. She was arrested and jailed for a week. In
court, she told the judge she’d acted in self-defence, and he found her not guilty. Her perpetrator was jailed for six months instead. The would-be rapist was
an Aboriginal man. “No way she would have gotten away with shooting a white ­assailant,” Lucashenko says. “My mother … was determined that her daughter would
have the chances she didn’t.” On leaving school in 1983 at the end of year 12, Lucashenko worked as a barmaid, a delivery truck driver and a support worker for
women out of prison. She used to be obsessed with karate and has a black belt in it. In 1990, she graduated from Griffith University with an honours degree in
public policy, having been the first in her immediate family to go to university.
</item 42>

<item 43>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
After years of getting along, some tension began to show in 2022. Spears and Federline both shared negative posts on social media about the other, which they
both deleted. Soon after, Preston and Jayden did a rare interview with ITV to discuss the situation.

While Jayden clarified that there was "no hate" involved towards anyone, he said that "it will take a lot of time and effort" to fix their relationship with
their mother. He clarified, "I 100% think this can be fixed. It's just going to take a lot of time and effort. I just want her to get better mentally. When she
gets better, I really want to see her again."

In a message directly to Spears, he said, "I love you a lot, I hope for the best for you. Maybe one day we can sit down like this and talk again."

LIKE WHAT YOU’RE READING?
</item 43>

<item 44>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.
</item 44>

<item 45>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
During an interview with Hoda Kotb for Today, which aired on 13 October, the actor confirmed that she and her partner have been living “completely separate
lives” for the past seven years. However, according to Jada, their choice to live in separate homes doesn’t mean that they’ll be getting divorced anytime soon.

“I made a promise that there will never be a reason for us to get a divorce,” she said. “We will work through whatever. And I just haven’t been able to break
that promise.”

When asked why she and Will didn’t “just amicably divorce,” Jada simply replied: “We don’t want to.”

She then explained how her family shaped that decision, as she and the King Richard star share two children - Jaden, 25, and Willow, 22. Will also has a
30-year-old son, Trey, who he shares with ex-wife Sheree Zampino. “We love our family and we love each other.
</item 45>

<item 46>
Blue Halloween buckets for autism: What they mean and why they're sparking controversy (FOX News - Lifestyle, 2023-10-31T20:13:41+00:00)
"The blue bucket can be a teaching tool for others; however, it should never be about others — it should always be about your child."

"In an ideal world, all adults would understand that not everyone communicates the same way, and that one's ability to communicate does not indicate one's
interest in candy," she said. "But, the reality is that not all adults know this."

CLICK HERE TO GET THE FOX NEWS APP

"It's exhausting teaching others to be inclusive," Coleman added. "So, yes, the blue bucket can be a teaching tool for others; however, it should never be about
others — it should always be about your child."

Madeline Farber contributed to this report.
</item 46>

<item 47>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
But the amount of time they’ve spent separated begs questions about all long-term relationships in general. When is it the time for couples to throw the towel
in? And when should separations ultimately become the end result that couples have tried to avoid: divorce.

While there’s never a simple answer to the question, relationship experts have discovered the signs that it’s time to keep fighting for the one you love, or
time to let them go.

Speaking to The Independent, New York City-based relationship expert Susan Winter highlighted the jarring differences between a separation and divorce. When it
comes to divorce, Hollywood couples can notably get involved in lengthy legal processes, from dividing financial assets to creating child custody agreements.
Although we don’t know the terms of Jada and Will’s separation, Winter believes that they’re showcasing a non-conventional relationship that’s not familiar to
the public. Will and Jada’s dynamic includes living in separate homes, but it could be a choice that’s for the better.
</item 47>

<item 48>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
She shares her eldest children, Ava and Deacon, with ex-husband Ryan Phillippe, and her youngest, Tennessee, with ex-husband Jim Toth.

In a recent interview on the "Good Inside with Dr. Becky" podcast, the Oscar-winner said she wants to see kids live with failure a little more.

"I see this a lot with parents – I don't know when we stopped letting our kids fail. Like I learned so much from the paper I didn't turn in or the demerits I
got, so I got detention. I was suspended from school," she recalled.

She added, "And my parents didn't say, ‘Uh, she didn't deserve that,' and take me out of school. They actually let me sit in it and feel uncomfortable. So, I
think learning from failure is actually a valuable tool that you can't take away from kids, right? You rob them if you don't let them sit in the discomfort of
the experience."
</item 48>

<item 49>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
These parents, said Keeley, "subconsciously compensate for a lack of self-worth and validate their role as a [parent] by creating too much busyness in their
life."

She said that "all that results from a mom who runs herself into the ground is just that: a mom who runs herself into the ground."

DELICIOUS CUT-OUT SUGAR COOKIES FOR CHRISTMAS: TRY THE SIMPLE RECIPE

She added, "If this is you, take a breath, put away the to-do list and ask yourself, ‘What would I have to believe in order to have a relaxed and joyful
Christmas?’ Maybe we can use that as a guide to sit back and relax."

4. Griping like the Grinch

Parents should be sure not to be "grinchy" around their kids amid holiday stress, Keeley said.

Children up to the age of 9 process information egocentrically, she said — which means if there’s a problem, they see themselves as the cause.
</item 49>

<item 50>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
For example, will the survey go to under 13 year olds? Since everyone needs to be at least 13 years old before they create an account, we want to be careful
about sharing findings that come back and point to under 13 year olds being bullied on the platform.”

In 2021, another, studying “child-adult sexual-related content/behavior/interactions” (!) said she was “not includ[ing] younger kids (10-12 yos) in this
research” even though there “are definitely kids this age on IG,” because she was “concerned about risks of disclosure since they aren’t supposed to be on IG at
all.”

Also in 2021, Meta instructed a third-party research company conducting a survey of preteens to remove any information indicating a survey subject was on
Instagram, so the “company won’t be made aware of under 13.”
</item 50>

<item 51>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“I feel as though Will has a tremendous love and commitment to Jada, as well as to the marriage and the family unit,” she said. “The pride that he takes in
defending his wife is extraordinary, almost to the detriment of his image, at times.”

Ultimately, the public will never fully know what “heavy-duty work” Will and Jada are doing in their relationship, and according to Winter, it’s not necessarily
any of our business. Still, she believes that their choice to separate is something fluid and outside of the traditional marriage design - an evolving model
that could also come to serve other couples. If Will and Jada have been spending seven years apart as a married couple, why would they decide to go back to
living in the same household?

“They’re in the rough with a machete, carving a new path for themselves,” Winter said. “Why would they say: ‘Hey, let’s go back to the house and just relax?’
</item 51>

<item 52>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
From there, things move faster but the kids were on board for things to move on.”

The mother specified that it was clear to her children that her husband “was not going to be filling the role of a dad in their lives”. She added that her
partner “was fine with that, as long as he was respected and they could work toward being a caring family”.

She continued to note that while she had a close relationship with Calla, it was “extremely tough” for her child to lose her father. She specified that while
her child has a good relationship with her stepfather, “it comes nowhere close to the bond she had with her dad”. She also shared that she continued to get
close to her daughter after she turned 12, before recalling a point in Calla’s childhood when she asked her mother to walk her down the aisle.
</item 52>

<item 53>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“So there are two possible outcomes. One, we are going to be together ‘til death, or two, I am dead.”

Speaking to The Independent, Lucas noted that while it’s important for both parties to be invested in their relationships, promising not to get a divorce could
spark feelings of resentment down the line. Fans aren’t fully aware of why Will and Jada made that promise to each other, but the dating coach believes that an
agreement like theirs could create a false narrative about a divorce.

Jada Pinkett Smith y Will Smith (Getty Images)

“Getting divorced shouldn’t feel like a taboo because people grow apart and that’s fine,” he explained. “People get stuck in relationships because of their
friends and family or societal pressure, and they worry about getting divorced. They think a divorce has a stigma to its name, which it doesn’t, and they push
aside the fact that they still essentially feel trapped with someone they don’t want to be with.”
</item 53>

<item 54>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Outsiders are now aware that Jada and Will have been separated since 2016, but the timeline of her relationship with Alsina is still fuzzy. Indeed, the terms of
Will and Jada’s separation are still unknown to the public and in some cases, having separate relationships can give couples the clarity they need about their
marriage.

“What’s pivotal here is, what was the agreement? Did we agree to separate to work on ourselves for the purpose of coming back together again?” Winter said. “Did
we agree to separate and just be alone for a while and not have that discussion? And in the second case, if you have an affair, sometimes it can bring you back
to the original person. You can get whatever you need to achieve or gain a different perspective, allowing you to decide if you want to come back or leave for
good.”
</item 54>

<item 55>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“It’s an LAT partnership, meaning ‘living apart together,’” Winter said. “While everybody is at a different point as to why they are doing this, the centre
point is the same: the preservation of the partnership is paramount. Separations can also be beneficial if both individuals are working on themselves and have
the mutual goal of reuniting, or making their relationship better. However, that doesn’t necessarily mean that their improved relationship will result in them
living together again.”

On the other hand, dating coach Jacob Lucas - who’s based in the UK - acknowledged that not all separations end in resolutions. When couples opt not to live
together anymore, it could create an emotional disconnect between them. If a lack of connection occurred before couples separated, that separation doesn’t
necessarily make the issues go away. In fact, Lucas believes that in these situations, couples could also be avoiding the inevitable breakup for the sake of
comfortability.
</item 55>

<item 56>
Emily Ratajkowski’s latest photo shoot sparks fury (The Independent - Life and Style, 2023-11-30T20:02:31+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Emily Ratajkowski’s latest photo shoot has sparked backlash among plus-size models and activists, who claim it promotes unhealthy narratives about body image.

The supermodel, 32, recently shared several images from her photo shoot with M Le Monde magazine. For the fashion editorial photographed by Oliver Hadlee
Pearch, Ratajkowski posed in a variety of looks featuring denim jeans and flowing tops. However, it was the second image that caught the attention of her
Instagram followers.
</item 56>

<item 57>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
Andrews and McDonald’s second book feels markedly different to their first, The Space Between. Published in 2020, it looked to explore the complications of life
in one’s 20s.

Advertisement

“We wrote it when we were so young,” reflects Andrews. “We truly had internalised this belief that for our words to be worthy of attention, they had to mine the
deepest, darkest events that have ever happened to us.” This feeling is one Andrews has spoken about before, referring to an essay she wrote for the book which
included details of her sexual assault. “I wish I’d just saved that for myself,” she says, adding that while she wouldn’t take back the essay, she would take
back some of the details. Zara McDonald wears Luuda shirt, $390, and shorts, $290. Converse shoes, $130, from Hype DC.
</item 57>

<item 58>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“It saddens me to hear that my ex husband has decided to discuss the relationship between me and my children … As we all know, raising teenage boys is never
easy for anyone,” she wrote. “It concerns me the fact that the reason is based on my Instagram … It was LONG before Instagram … I gave them everything … Only
one word: HURTFUL.”

Asghari went on to accuse Federline of using their sons to “vilify” his wife, and claimed her ex-husband’s “gravy train will end soon, which probably explains
the timing of these hurtful statements.”

“There is no validity to his statement regarding the kids distancing themselves and it is irresponsible to make that statement publicly,” Asghari wrote on
Instagram.

Following Federline’s comments, Spears’s attorney Mathew Rosengart said in a statement obtained by Variety that Federline has created “legal issues” for
himself, after posting private videos of Spears sternly speaking with her children without her consent.
</item 58>

<item 59>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
In an interview with People on 10 October, Jada shared that during the 2022 Academy Awards - when Will slapped Chris Rock after he cracked a joke about Jada’s
appearance - the pair had actually been separated for six years. “We’ve been doing some really heavy-duty work together,” she explained, while promoting her new
memoir, Worthy. “We just got deep love for each other and we are going to figure out what that looks like for us.”

She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep
sense of self-acceptance”. Although the married couple are not living together, she insisted that their marriage has never been an open one, as she and Will
have “eliminated the chance of betrayal” while going on to live their “separate lives”.
</item 59>

<item 60>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Spears is also the mother of two children – Sean, 18, and Jayden, 17 – who she shares with her second ex-husband, Kevin Federline.

According to a press release for The Woman in Me, Spears’ memoir is “a brave and astonishingly moving story about freedom, fame, motherhood, survival, faith,
and hope”. Spears’ book is also expected to cover the high-profile legal battle to end her conservatorship, which was dissolved in November 2021 after 13 years.

Throughout her time in the spotlight, specifically during the ‘90s and 2000s, Spears has been in some heavily publicised romances. Unsurprisingly, fans are
anticipating that Spears will detail some of her relationships in The Woman in Me, which is set for release on 24 October.

From Justin Timberlake to her now ex-husband Sam Asghari, here’s everything we know about Britney Spears’ relationships and rumoured romances.
</item 60>

<item 61>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
Michelle Andrews (right) wears Venroy shirt, $280, and pants, $280. Credit: Hugh Stewart

Advertisement

Only five per cent of the letters submitted to the Ask Shameless inbox are published. But after reading hundreds of requests for advice in the past three years,
Andrews and McDonald are in touch with the most prevalent issues plaguing women today. One of those, says Andrews, is that “a lot of young women do not receive
the basic level of affection, attention or love from their male partners”. While there are questions the pair don’t feel qualified to answer yet – such as when
to have kids – there’s also a kind of magic in the fact that Shameless listeners have access to McDonald and Andrews while they’re still in the thick of these
struggles themselves. Loading Building a brand on trust is both a big achievement and a risk. How worried are they about getting it wrong? Turns out, not very.
</item 61>

<item 62>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce’s previous comments about having children have sparked controversy amid his ongoing romance with Taylor Swift.

The Kansas City Chiefs player, 34, first made the remarks about starting his own family during a February 2023 episode of his and his brother Jason Kelce’s
podcast, New Heights. The episode resurfaced after a source claimed to Yahoo Entertainment that Swift and Travis Kelce “have talked about kids” within the next
year.
</item 62>

<item 63>
Martin Scorsese gets quizzed by his daughter on internet slang (The Independent - Life and Style, 2023-10-10T14:00:16+00:00)
This isn’t the first time Scorsese has made a special appearance in one of his daughter’s TikTok videos. In April 2021, Francesca went viral when she asked her
father to identify “feminine products”, such as an eyelash curler, bobby pins, a hair donut and nipple pasties.
</item 63>

<item 64>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

In her forthcoming memoir, The Woman in Me, Britney Spears is expected to reflect on some of the most pivotal moments in her life, including her relationships.

Over the years, the pop star’s love life has continued to make headlines. Most recently, her ex-husband Sam Asghari filed for divorce after only 14 months of
marriage.
</item 64>

<item 65>
Emily Ratajkowski’s latest photo shoot sparks fury (The Independent - Life and Style, 2023-11-30T20:02:31+00:00)
While the body positivity movement has made waves over the past 10 years, this year in fashion was especially devoid of inclusivity or plus size models.
According to Vogue’s Business Size Inclusivity Report, only 0.6 per cent of looks shown throughout the autumn/winter 2023 fashion season were modelled by
plus-size people.

Despite the fact that the plus-size clothing market is expected to reach $288bn this year, according to Future Market Insights, advocates have maintained that
they’re unable to purchase clothing that’s both sustainable and inclusive. “I have to wear fast fashion and high street to major red carpet moments, and that’s
great, but I want the option to wear high end items of my choosing and it’s not available,” Holliday told Elle UK in 2019. “But it’s either that or I run around
naked. Plus-size fashion is not there yet.”

The Independent has contacted representatives for Ratajkowski and M Le Monde for comment.
</item 65>

<item 66>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
“We love our family and we love each other. It’s more of a life partnership," Jada explained. "Now, 10 years from now, Hoda, who knows?”

During a 2018 episode of her Facebook Watch series, Red Table Talk, Jada doubled down on her stance about divorce, as she recalled the the hard conversation she
had with Will about the topic before they got married.

(Getty Images)

“I told Will from the gate, I said let me tell you something: ‘If you marry me, know this: we’re gonna be together. We’re going to be under the same roof,’” she
said. “For me personally, I’m not mature enough to have a divorce, I’m just not. I don’t think I would ever be mature enough, I don’t.”
</item 66>

<item 67>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Will Smith y Jada Pinkett Smith (Getty Images)

According to Lucas, some of the controversy surrounding Will and Jada’s relationship ties back to the 2022 Academy Awards - when the King Richard star jumped
onto the stage and struck Rock across the face after the comedian made a joke about Jada’s shaved head, which is a result of her alopecia. “Keep my wife’s name
out of your f***ing mouth,” Will yelled. Now that it’s been revealed the couple was separated at the time, Lucas questioned Will’s behaviour and word choice
during the awards ceremony.

“That’s a weird reaction to have somebody who you’ve been broken up with for six years, you know? So, were they really broken up or is Jada saying that as an
attention-seeking thing?” he asked.

However, Winter still believes the controversial incident speaks different volumes about the work Will has put into his relationship.
</item 67>

<item 68>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Speaking to the crowd, Will said that Jada’s positive relationship with Zampino allowed all three of his children the chance to grow up together.

“Can you show up and love somebody for the rest of your life, even when you don’t agree with them?” he asked. “Jada is the best friend I have ever had on this
planet and I am going to show up for her and support her for the rest of my life.”

This isn’t the first time that Will has opened up about his relationship, as it’s continued to make headlines. Earlier this week, he appeared to mock the
publicity storm surrounding him and Jada in a video titled “Official statement” shared on Instagram. “So, here’s the thing. My opinion is... uh…” the actor
said, before sneezing into the crook of his arm. As he sneezed, the video comically zoomed out to show several different landscapes including lakes and
mountains.
</item 68>

<item 69>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“I told Will from the gate, I said let me tell you something: ‘If you marry me, know this: we’re gonna be together. We’re going to be under the same roof,’” she
said during a 2018 episode of Red Table Talk. “For me personally, I’m not mature enough to have a divorce, I’m just not. I don’t think I would ever be mature
enough, I don’t.”

Will - who divorced his first wife, Sheree Zampino, in 1994 - has also shared similar sentiments about making his marriage with Jada work. Back in 2006, the
Fresh Prince of Bel-Air star first explained to MTV News that divorce wasn’t an option for him and his second wife. “With Jada, I stood up in front of God and
said: ‘Til death do us part,’” he said at the time. “So there are two possible outcomes.
</item 69>

<item 70>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
"And so when he got the shower on, and he was whistling and singing in there and thinking Daddy's not around, all the sudden the ice-cold shower came on. And
all of a sudden, we heard downstairs the scream," Schwarzenegger said.

The former California governor also recently issued a warning that if young people are over-pampered, this nation may be "creating a generation of wimps and
weak people."

"You can only strengthen your character and become a really strong person inside if you have resistance, if you fail, if you get up again and if you work hard,"
Schwarzenegger said on "The Howard Stern Show."

ARNOLD SCHWARZENEGGER'S BODYBUILDER SON IS SPITTING IMAGE OF FAMOUS DAD

Reese Witherspoon

Reese Witherspoon is a mom of three.
</item 70>

<item 71>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
I don’t think I would ever be mature enough, I don’t.”

The Fresh Prince of Bel-Air star has also shared similar sentiments, as he said during an appearance on the Ellen Degeneres Show that “divorce just can’t be an
option” for him and Jada. “It’s really that simple. And I think that’s the problem with LA - there are so many options. So a huge part of the success for [Jada]
and I is that we just removed the other options.”

Last week, Jada made headlines when she revealed that at the time of the 2022 Academy Awards, when Will slapped Chris Rock, she and her husband had been
separated for six years. “We’ve been doing some really heavy-duty work together,” she said during an interview with People. “We just got deep love for each
other and we are going to figure out what that looks like for us.”
</item 71>

<item 72>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Elizabeth comes with an open mind, ready to inhabit Gracie and fully understand her. And yet, across Todd Haynes’ May December, we see how nothing is as simple
as Elizabeth — or the audience — thinks. Gracie is always more than meets the eye: more emotional, more aware, more monstrous. And throughout the movie, Haynes
expertly teases out the two women’s mysteries as they each try to craft an identity around the other.

Haynes’ film certainly has callbacks to other, classic movies about selfhood. But in a way, May December feels most in line with something like The Rehearsal,
which also has a lot on its mind about how blurry the line between real life and performance can be, and particularly how traumatic it is to ask kids to behave
like adults. It’s something Melton’s Joe is tragically caught in the middle of, and something May December always has on its mind, even when it feels like it’s
about other things.
</item 72>

<item 73>
Emily Ratajkowski’s latest photo shoot sparks fury (The Independent - Life and Style, 2023-11-30T20:02:31+00:00)
Plus-size models have long advocated for more inclusive sizing - both in stores and on the runway. However, luxury brands and labels often avoid creating
plus-size fashion over claims that it’s too expensive to make. “Designers won’t make plus sizes unless it’s for a photo opp where a thin person can be quirky,”
another person pointed out.

Many people were especially surprised by the “distasteful” photo shoot considering Ratajkowski has published a book of essays about feminist empowerment and the
objectification of women’s bodies, titled My Body. “Not sure how you think wearing oversized jeans made for a much larger person and accentuating your smallness
promotes healthy body image like you preach in your books/social media,” an Instagram user said.

Since she shared snaps from the photo shoot over the weekend, Ratajkowski has not yet commented on the backlash. Despite the controversial editorial, the model
has previously spoken out against body-shaming. In her book of essays, Ratajkowski discusses bodily autonomy and exploitation in the modelling industry.
</item 73>

<item 74>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
1997: Will and Jada get engaged and married

Will popped the question to Jada in November 1997 before getting married just one month later in Baltimore.

Jada wasn’t initially sold on the idea of marriage. She told People in 2019: “I never wanted to get married. But my mother was like: ‘You have to get married’ -
she’s so old-school - and Will wanted a family. So I said: ‘All right, maybe it’s something I should do.’”

1998: Their first child is born

Jada and Will’s first child together, son Jayden, was born in July 1998.

2000: Their second child was born

Jada and Will welcomed daughter Willow in 2000.
</item 74>

<item 75>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
She has my support till death and it feels so good to get to that space.”

2020: August Alsina claims he had an affair with Jada

August Alsina and Jada were introduced in 2015, but it wasn’t until 2020 that he claimed that he and Jada had dated while she was married and that her and
Will’s marriage was open during that time.

Alsina said Will gave him and Jada his “blessing”, adding: “I actually sat down with Will and had a conversation due to the transformation from their marriage
to life partnership that they’ve spoken on several times and it not involving romanticism.”

Alsina added he “totally gave myself to that relationship for years of my life and I truly and really, really deeply loved and have a ton of love for her.”

The same day Alsina’s claims surfaced Jada’s reps said they were “absolutely not true”.
</item 75>

<item 76>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed why she and her husband Will Smith never signed a prenuptial agreement before getting married.

The former talk show host, 52, spoke candidly about her husband during a recent interview with Parade. While discussing her new memoir, Worthy, Jada revealed
that she and Will didn’t sign a prenup when they were married in 1997 because they knew divorce wasn’t in the cards for them.
</item 76>

<item 77>
Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked (The Independent - Life and Style, 2023-12-06T16:45:51+00:00)
In a statement to The Independent, a Roblox spokesperson revealed the company is looking into Gellar’s complaint. “We are looking into this case and we reached
out directly to the user to help. We are committed to ensuring that our users have a positive experience on the platform,” they said. “In partnership with
safety groups, we’ve developed tools and resources like our Parent & Caregiver Guide and other helpful resources for parents to have conversations with their
kids and teens about staying safe, spending controls and protecting their account. Our support team works with parents and users directly 24/7 and can be
contacted with questions here,” they added, including a link to its support page.

The Scooby-Doo star isn’t the only celebrity to publicly call out Roblox’s parental controls.
</item 77>

<item 78>
Emily Ratajkowski’s latest photo shoot sparks fury (The Independent - Life and Style, 2023-11-30T20:02:31+00:00)
However, it was the second image that caught the attention of her Instagram followers.

In the photo, Ratajkowski is seen wearing a pair of low-rise, wide-leg denim jeans double her size. She showed off her toned figure as the oversized pair of
jeans fell at her feet. As she held up the waistband of the jeans with one arm, Ratajkowski stood in one leg of the pair of jeans to highlight the noticeable
size gap.

The photo shoot - which was styled by Charlotte Collet and creative directed by Jean-Baptiste Talbourdet-Napoleone - instantly sparked backlash after
Ratajkowski shared the images to Instagram. In her comments section, thousands of users criticised both Ratajkowski and the team behind the photo shoot for the
“out of touch” image and claimed the photo shoot was “fatphobic”.
</item 78>

<item 79>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
His recent comments come after Jada made major revelations about their marriage in her memoir, including how she and Will have been separated for seven years.

During the event in Baltimore, Will began by speaking to the crowd about his love and admiration for his partner, who he married in 1997. “I just really wanted
to come out and just be here and hold it down for you the way you have held it down for me,” he said during the event, as reported by The Baltimore Sun.

The King Richard star went on to describe the challenges and achievements he’s had throughout his marriage, adding: “We have had a very, very long and
tumultuous [relationship]. We call it ‘brutiful.’ It was brutal and beautiful at the same time."

At another point during the event, Will referred to his union with Jada as a “sloppy public experiment in unconditional love,” according to the Baltimore
Banner.
</item 79>

<item 80>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
However, during an appearance on the Today show on 16 October, Jada clarified to Hoda Kotb that she and the Suicide Squad actor are “working hard” and are
“concentrating on healing” to get their relationship to a better place.

“There’s no finding another great love, and I think that’s the point,” she said. “It’s like we are in a place now that we are in a deep, healing space. And we
are really concentrating on healing the relationship between us.”

The Girls Trip star continued: “There’s no divorce on paper. We really have been working hard. That’s the whole thing. We are working very hard at bringing our
relationship together. Back to a life partnership.”
</item 80>

<item 81>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
The Santa bribe turns the focus of Christmas into "selfish gain on the part of your children," Keeley added.

"Love your kids enough to learn how to manage their behavior appropriately and teach them self-discipline."

"It’s also lazy parenting, which never pays off in the long run," she said. "Being a mom is a professional career and deserves to be treated accordingly."

She added, "Love your kids enough to learn how to manage their behavior appropriately and teach them self-discipline."

3. Becoming a ‘holiday martyr’

It’s important to keep the holidays well-rounded, Keeley said.

5 HOLIDAY DECORATING MISTAKES THAT COULD REDUCE YOUR CHEER THIS YEAR

She called out "martyr moms" who are at their "peak performance" during Christmastime, putting everyone else before themselves.

"They wear themselves out making sure everyone else has the perfect Christmas, but never put themselves on the receiving end," she said.
</item 81>

<item 82>
Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked (The Independent - Life and Style, 2023-12-06T16:45:51+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Sarah Michelle Gellar has taken umbrage with Roblox after her son’s account was hacked, forcing her to cancel her credit card.

The Buffy the Vampire Slayer star, 46, took to Instagram on 4 December after her son’s Roblox account was allegedly hacked.
</item 82>

<item 83>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
The Independent has contacted u/PopShort7064 for comment.
</item 83>

<item 84>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance.”

On 16 October, Will broke his silence on his wife’s comments during an interview with the New York Times. The outlet noted that he said his partner’s memoir
“kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” Will added.
</item 84>

<item 85>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.
</item 85>

<item 86>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
says McDonald. “How do we do this and have a family if the whole business is relying on us showing up together at the same time?”

Advertisement

The only way to ensure it wouldn’t be an issue was to create a contingency plan. In their case, that plan involved expanding Shameless Media by growing the team
and up-skilling junior employees. While neither Andrews nor McDonald sees kids in her near future, they found it reassuring to work towards a solution.
Regardless of the topic of conversation, whether it be personal or work-focused, the two try to be completely honest with each other. “We realised with every
honest conversation we had, our relationship was so much better – it was far more intimate,” says McDonald. The relationship McDonald and Andrews have with
Shameless listeners is also based on trust. Their podcast episodes are meticulously well-researched, book recommendations are solid and the advice given via
their Ask Shameless newsletter aims to always hit the mark.
</item 86>

<item 87>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
But now, as he's preparing for a return to television and focusing so much on parenting, things seem to be looking up.

Fox News Digital takes a look at other celebrity fathers who have raised or are raising their children on their own.

John Travolta

John Travolta and Kelly Preston were married in 1991, and they had three children together: Jett, who died at 16 in 2009; Ella, 22; and Benjamin, 13. Preston
passed away in 2020 after battling breast cancer.

In 2021, Travolta appeared on Kevin Hart's interview show "Hart to Heart" and shared an intimate conversation he had with Benjamin about the loss of his mother,
recalling, "(Ben) said to me once, 'Because mom passed away, I'm afraid you're going to.' I said, 'Well, it's a very different thing.' And I went through the
differences about my longevity and her limited life."
</item 87>

<item 88>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
SARAH MICHELLE GELLAR, FREDDIE PRINZE JR REVEAL SECRETS BEHIND THEIR LONG-LASTING MARRIAGE

"They know they can’t break [those rules]," he said. "Friendships come and go. I tell my kids all the time that family is forever. So, we have rules, and
they’ve got to follow [them]."

Gellar told Us Weekly in 2020 that as parents their "expectations are a little higher than most."

"I think [our children] know what the rules are," she said at the time. "We have [family dinner] as many nights as we can. We have no phones at the table. We
sit, we all have dinner together."

Ashton Kutcher and Mila Kunis

Ashton Kutcher and Mila Kunis have been married since 2015 and share two children, daughter Wyatt, 9, and son Dmitri, 6.
</item 88>

<item 89>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
You rob them if you don't let them sit in the discomfort of the experience."

REESE WITHERSPOON AND EX JIM TOTH SETTLE DIVORCE WITH KEY STIPULATION ABOUT RAISING THEIR SON

The "Walk the Line" star remembered comforting her daughter when she was younger after she lost a basketball game – but not shying away from the lesson.

Witherspoon recalled saying, "'Yeah, I know. I know that probably feels really bad,'" but then adding, "'You know what also? Maybe you're not good at
basketball?'"

Her daughter was slightly upset in the moment, but Witherspoon continued, "It's actually really important to learn what you're not good at."

CLICK HERE TO SIGN UP FOR THE ENTERTAINMENT NEWSLETTER

Daniel Craig

James Bond star Daniel Craig shares one daughter with wife Rachel Weisz and has an older daughter from his previous marriage to Fiona Loudon.
</item 89>

<item 90>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Here’s a timeline of their relationship.

1994: Will and Jada meet on the set of The Fresh Prince of Bel-Air

Will and Jada met after she auditioned for the role of Will’s girlfriend on The Fresh Prince of Bel-Air. While she did not get the part, the pair hit it off -
despite Will being married to Sheree Zampino at the time, with who he shares son Trey.

“I had a realisation I wasn’t with the person I was supposed to be with,” he told Jada’s Red Table Talk in 2018. “I was sitting in a [bathroom] stall and I was
crying and laughing uncontrollably, and I knew [Jada] was the woman I was supposed to be with.”

Will and Jada in 2000 (Getty)

1995: The pair officially begin dating

It wasn’t until 1995 that Will and Jada began seriously dating after Will finalised his divorce from Zampino.
</item 90>

<item 91>
Emily Ratajkowski’s latest photo shoot sparks fury (The Independent - Life and Style, 2023-11-30T20:02:31+00:00)
“Every woman I know - doesn’t matter what they look like, or if they’ve commodified their image or not - knows what it feels like to be looked at, to be
rejected, to get attention for how they look,” she told The Independent in 2021.

Back in 2019, Ratajkowski defended her friend against body-shaming trolls after their bodies were compared on Instagram. The model shared a bikini-clad post to
promote her swimwear line, Inamorata, but several users accused Ratajkowski of sharing the photograph to make her body look “better”.

In response, Ratajkowski commented: “I love my friend’s body and both her and I think she looks great here! And I’m proud she’s rocking my suits. All these
haters are crazy. Just because you’re used to seeing one body type on the internet doesn’t mean that that’s the only kind that should be considered
‘beautiful’.”
</item 91>

<item 92>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
In a recent interview with People while promoting his new book, he recalled a moment when his eldest daughter, Katherine, brought over her 3-year-old daughter,
Lyla, and saw the lesson he passed down.

"Katherine comes over with Lyla, and she says, 'Lyla, I told you already not to put the shoes there. Keep your shoes on, or you put them away, but you don't
leave them there by the stand in front of the fireplace because you know what Daddy did? When I left my shoes there twice? The third time, he burned them in
front of me and I cried,'" he said.

Schwarzenegger said the story was "absolutely" true, and he took pride in knowing that Katherine "uses the same methods, which she cried over, and she
complained about" with Lyla as well as her other daughter, Eloise, both of whom she shares with husband Chris Pratt.
</item 92>

<item 93>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
However, according to the Reddit user, her husband didn’t take the news as expected. She explained that he asked her a series of questions, with the first one
being if she asked Calla if both her mother and stepfather could walk her down the aisle. The woman noted that she didn’t ask her daughter that.

“Then he asked if she ever considered asking him, and I said I could not answer that for her. He asked me if I thought of him when I said yes,” she continued.
“He asked whether I gave any thought to all he has done for Calla, for both kids, and the fact he’s still not looked upon as a fatherly figure all these years
later. I told him I did not think of him when asked because I was overjoyed. He told me I should have given him a lot more consideration and I should have tried
to compromise with my daughter.”
</item 93>

<item 94>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Less than two weeks later, Jada and Will appeared on an episode of Jada’s Red Table Talk where Jada confirmed that she did have a relationship with Alsina,
saying that she and Will were separated at the time.

“We decided that we were going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,”
Jada said at the time.

“From there, as time went on, I got into a different kind of entanglement with August... Yes, it was a relationship absolutely.”

Jada also said that Will didn’t give permission, as the only person who could give her permission was herself.

2021: Jada discusses her and Will’s sex life, Will’s memoir is released

In another episode of Red Table Talk in October 2021, Jada said it “takes work” to maintain a spark in the bedroom.

“It’s hard... the thing Will and I talk about a lot is the journey.
</item 94>

<item 95>
Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure' (FOX News - Entertainment, 2023-10-10T08:00:40+00:00)
Hollywood is a tough place to raise kids, prompting some celebrities to take a tough-love approach to parenting.

Stars like Arnold Schwarzenegger, who admitted to being like a "drill instructor" when his kids were young, and Reese Witherspoon, who recently said she wants
kids to deal with failure more directly, have shared their harsher parenting styles in interviews.

Others, like Ashton Kutcher and Mila Kunis and Kristen Bell and Dax Shepard, are focused on keeping their kids grateful for growing up in famous families with
wealth and privilege at their fingertips.

Here's a look at a few celebrity parents focusing on firmer parenting.

Arnold Schwarzenegger

Arnold Schwarzenegger, who shares four children with ex-wife Maria Shriver and also shares a son with Mildred Baena, the family's former housekeeper, has a
disciplined approach to parenting, recently admitting he had a "drill instructor" approach when his kids were young.
</item 95>

<item 96>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Longtime Hollywood couple Will Smith and Jada Pinkett Smith sent shockwaves around the world this week when the Red Table Talk host revealed that she and the
actor have been separated since 2016. Both Jada and Will have previously doubled down on their stance against getting a divorce, but her recent reveal about
their lengthy separation poses a new question: When should couples just get a divorce?
</item 96>

<item 97>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“We’ve been doing some really heavy-duty work together,” Jada said. “We just got deep love for each other and we are going to figure out what that looks like
for us.”

The pair have been married since 1997 and share son Jaden, 25, and daughter Willow, 22. The Fresh Prince of Bel-Air star also has a son Trey, 30, whom he shares
with his first wife, Sheree Zampino.

As for her children, Jada praised them for how they’ve helped her grow. “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance,”
she told the outlet.

Jada and Will have often opened up about their relationship over the years, and have had very public ups and downs throughout their marriage. Here’s a timeline
of their relationship.
</item 97>

<item 98>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Timberlake proves to be the last person you’d call in a crisis. “At some point he thought maybe music would help, so he got his guitar and he lay there with me,
strumming it,” she writes. Almost as embarrassing is husband Kevin Federline’s attempt to start a music career. “He really thought he was a rapper now,” she
writes, with the bafflement of someone who has never known pretension. “Bless his heart – because he did take it so seriously.”

Motherhood leaves her vulnerable

When Britney gets pregnant with her first son, she initially thinks it might protect her: “I wanted everyone to stay away: stand back! There’s a baby here!”
Inevitably, it doesn’t work out that way. The paparazzi become aggressive, particularly once she has her kids in quick succession – becoming pregnant a second
time three months after giving birth to her first.
</item 98>

<item 99>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It may be over a decade since Britney Spears and Kevin Federline ended their marriage, but the former couple continue to have their ups and downs when it comes
to co-parenting their two teenage sons, Sean Preston and Jayden.

Ahead of the release of the pop star’s memoir, The Woman in Me, which is due for release on 24 October, fans are anticipating Spears will share new details
about her relationship with her second husband, Federline.
</item 99>

<item 100>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
"I'm thinking I found my birth parents and my half-siblings," Bazella said. "It's like the story was kind of over in 1983."

PARENTS WELCOME 14-POUND BABY, THE LARGEST ON RECORD SINCE 2010: 'EVERYBODY WAS MAKING BETS'

Adopted at nine months old, Bazella had been raised by kind and loving parents, the Petersons, in Edina, Minnesota.

When she was a child, Bazella learned through her adoptive parents that she was adopted in 1961, nearly one year after being born at Booth Memorial Hospital —
now known as the Booth Brown House Youth Shelter — in St. Paul, Minnesota, to 18-year-old parents who were unmarried.

TRENDIEST BABY NAMES OF 2023 REVEALED AS LIAM IS KNOCKED OFF NO.
</item 100>
=== END ITEM id=59ee0b2dedd2e7db ===

=== ITEM id=015f395b2c3b161e ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answer right back and sack Dobbs.
  Fact 0.1: Sweat and Brisker are in there to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:

=== END ITEM id=015f395b2c3b161e ===
