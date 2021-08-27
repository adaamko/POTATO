# HASOC 2020 training data

## TASK1

### Rule: Stupid

Inconsistent labeling in TASK1, examples:

If the following is labeled as HOF

    Sentence: RT it’s my birth month! go crazy 🤪🤪 go stupid 🤪🤪🥳

    Gold label: HOF

    Sentence: RT This is all so stupid

    Gold label: HOF

Then the following is why not:

    Sentence: it feels so stupid and irrational crying about something i more or less expected would happen

    Gold label: NOT

    Sentence: The stupid, it hurts

    Gold label: NOT

### Rule: Damn

True positives:

    Sentence: RT damn, i’m missing you again

    Gold label: HOF

    Sentence: damn

    Gold label: HOF

    Sentence: RT When you’re on your 5th “damn that’s crazy” and they still haven’t stopped talking https://t.co/T82qTo45d9

    Gold label: HOF

False positives:

    Sentence: Damn Courtnel, lob your boy an invite sometime.

    Gold label: NOT

    Sentence: Damn are you serious!!!!! I just followed you this morning. I left mine at the gym earlier today and now I… https://t.co/bvIv0TiABT

    Gold label: NOT

    Sentence: Damn it

    Gold label: NOT

    Sentence: damn is it on microwaves?

    Gold label: NOT

### Rule: Sick

True positives:

    Sentence: RT I am so SICK of Republicans endorsing positions simply to appease the media and the left The media loves Republicans th…

    Gold label: HOF

    Sentence: Truly sick.

    Gold label: HOF

False positives:

    Sentence: I’m so sick of this. Why are they still in a position of power over our Country and Legal Citizens. This is insani… https://t.co/kRhK4Oy6jq

    Gold label: NOT

    Sentence: RT I’m so sick of being sick and tired. The only thing stopping us from saving 40,000 lives annually isn't people caring ab…

    Gold label: NOT

### Rule: (u_1 / damn :mode (u_2 / .*))

True positives:

    Sentence: damn ok

    Gold label: HOF

    Sentence: damn

    Gold label: HOF

    Sentence: Well damn!

    Gold label: HOF

    Sentence: RT damn, i’m missing you again

    Gold label: HOF

    Sentence: damn

    Gold label: HOF


### Rule: moron

True positives:

    Sentence: His motive is to run against Trump. What moron!

    Gold label: HOF

False positives:
    Sentence: What a jerk, you moron that could have been your mother, grandmother, Aunt. I hope they keep in jail.

    Gold label: NOT

    Sentence: RT Both the Attorney General William Barr and President Trump now have the right to sue the ignorant and willfully moronic…

    Gold label: NOT

### Rule: idiot

False positives:

    Sentence: Ok but all you ugly racist idiots need to stop. My god no one asked for you opinion, much less is it needed. If you… https://t.co/XBae5w7jxt

    Gold label: NOT


    Sentence: RT you were supposed to double text me idiot

    Gold label: NOT


    Sentence: A bad deal was totally ruled out by the same vile idiot. The EU and the rest of the world were going… https://t.co/85mVJjbRKo

    Gold label: NOT

### Rule: niggas

False positives:

    Sentence: RT Moral of the story, Y’all will spend that $25 when niggas is poppin but won’t spend that $25 when niggas coming up. Y’al…

    Gold label: NOT

## TASK2

    Sentence: Nobody is hating on you bitch, in fact NOBODY gives a damn about what you do or don’t do 😊

    Gold label: PRFN

    Sentence: Go live in India bitch

    Gold label: PRFN

    Sentence: The fact that these niggas think it’s cool just to beat their dicks to women IN PUBLIC....WITHOUT THEIR CONSENT!!! Is fucking Insane.

    Gold label: PRFN

    Sentence: Niggas be like I never worked a 9 to 5 Well shit when y’all hiring cause I’m tryna have the wonder bread to

    Gold label: OFFN


# HASOC 2019 training data

## TASK1

### ass:

    Sentence: [USER] FutzPuck warmongeringAss dickhead ZEROCredibility WouldNotKnowTheTruthIfIfWereSlappedUpsideYourStupidFace HaHaHaTrumpPickedAWomanForAirForceSecretary NOTYOULOL YouSuck VoteBlueNotYou lol [USER] Knows You Are A PUTZ! BarbaraBennetNotYouFutzpuck [URL]

    Gold label: NOT

## shit:

    Sentence: Now he's trying to walk back what he said. Now he's saying of course you give it to the FBI. He's such a lying piece of shit. HappyJohnMcCainDay JohnMcCainDayJune14th TrumpIsATraitor TrumpLiesEveryTimeHeSpeaks

    Gold label: NOT


    Sentence: Tell someone who gives a shit. Murderer [URL]

    Gold label: NOT


    Sentence: [USER] [USER] [USER] talking shit tweet at end off tournament when were holding the trophy dickhead

    Gold label: NOT


    Sentence: [USER] FuckTrump who gives a shit what he does! June14thJohnMcCainDay

    Gold label: NOT

    Sentence: it's a sure bet that trump jr. has not read muellerreport, thus has no idea the size of the pile of shit he'll have to dig himself out from.under during senate intel hearing today. fucktrump

    Gold label: NOT

## rape:

    Sentence: [USER] Literally just called the majority of our population losers. SenileAF StableGenius racist rapist pervert treason Resist UnstablePresident 25thAmendmentNow

    Gold label: NOT

    Sentence: Where are all liberals? Liberals rapist [URL]

    Gold label: NOT

## fucktrump

There are lots of weird annotations regarding the #fucktrump hashtag.

False positives (there are around 50 of them):

    Sentence: Homophobia and transphobia are CRIMES in 43 countries. Yet here in America, under the Trump administration, it's seemingly encouraged. We need CHANGE. VOTE BLUE FOR 2020. FuckTrump

    Gold label: NOT

    Sentence: Trump and Bibi Are Evil Twinsies. Both are Autocratic Dictator wannabes who only care about themselves!! FuckTrump ImprisonTrumpNow [URL]

    Gold label: NOT

    Sentence: [USER] It's so satisfying yet horrifying.fuckTrump

    Gold label: NOT

    False positives: 47

    Sentence: [USER] This comment section is filled with Russian bots and NASCAR fans FuckTrump

    Gold label: NOT

There are also lots of true positives

    Sentence: Trump's Electoral Shenanigans Are Getting Worse Resist ResistTrump FuckTrump [URL]

    Gold label: HOF

    Sentence: FuckTrump And retired porn star Melania too. [URL]

    Gold label: HOF

    Sentence: Wasn't Nixon guilty of 'opposition research', I mean along with everything else? FuckTrump ImpeachTrump ImpeachTheMF ImpeachTrumpNow ImpeachTheMF

    Gold label: HOF